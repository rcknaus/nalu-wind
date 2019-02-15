/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/MomentumCoriolisSrcHOElemKernel.h>
#include <kernel/TensorProductCVFEMScalarBDF2TimeDerivative.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>

#include <element_promotion/ElementDescription.h>

#include <kernel/Kernel.h>
#include <SolutionOptions.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <AlgTraits.h>
#include <TimeIntegrator.h>

#include <CoriolisSrc.h>


// template and scratch space
#include <BuildTemplates.h>
#include <ScratchViewsHO.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <Teuchos_BLAS.hpp>

// topology
#include <stk_topology/topology.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

template<class AlgTraits>
MomentumCoriolisSrcHOElemKernel<AlgTraits>::MomentumCoriolisSrcHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    rhoRef_(solnOpts.referenceDensity_),
    cor_(solnOpts)
{
  ThrowRequire(solnOpts.gravity_.size() == 3u);

  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_coordinates_field(*coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);

  density_ = &meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
    ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*density_, 1);

  velocity_ = &meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity")
    ->field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*velocity_, AlgTraits::nDim_);
}
//--------------------------------------------------------------------------

Kokkos::Array<DoubleType, 3> compute_coriolis_src_vector(const CoriolisSrc& cor, const DoubleType* vel)
{
  const DoubleType ue = cor.eastVector_[XH] * vel[XH] + cor.eastVector_[YH] * vel[YH] + cor.eastVector_[ZH] * vel[ZH];
  const DoubleType un = cor.northVector_[XH] * vel[XH] + cor.northVector_[YH] * vel[YH] + cor.northVector_[ZH] * vel[ZH];
  const DoubleType uu = cor.upVector_[XH] * vel[XH] + cor.upVector_[YH] * vel[YH] + cor.upVector_[ZH] * vel[ZH];

  const DoubleType ae = +cor.corfac_ * (un * cor.sinphi_ - uu * cor.cosphi_);
  const DoubleType an = -cor.corfac_ * ue * cor.sinphi_;
  const DoubleType au = +cor.corfac_ * ue * cor.cosphi_;

  return Kokkos::Array<DoubleType, 3>{{
    ae*cor.eastVector_[XH] + an*cor.northVector_[XH] + au*cor.upVector_[XH],
    ae*cor.eastVector_[YH] + an*cor.northVector_[YH] + au*cor.upVector_[YH],
    ae*cor.eastVector_[ZH] + an*cor.northVector_[ZH] + au*cor.upVector_[ZH]
  }};
}
//--------------------------------------------------------------------------
template <class AlgTraits> void
MomentumCoriolisSrcHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>&  lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  nodal_scalar_workview l_vol(0);
  auto& vol = l_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);

  auto rho = scratchViews.get_scratch_view<nodal_scalar_view>(*density_);
  auto velocity = scratchViews.get_scratch_view<nodal_vector_view>(*velocity_);


  nodal_vector_workview work_coriolis_src;
  auto& coriolis_src = work_coriolis_src.view();
  constexpr int n1D = AlgTraits::nodes1D_;

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
       auto vec = compute_coriolis_src_vector(cor_, &velocity(k,j,i,0));
       for (int d = 0; d < 3; ++d) {
         coriolis_src(k, j, i, d) = vec[d] * vol(k, j, i);
       }
      }
    }
  }
  nodal_vector_view v_rhs(rhs.data());
  ops_.volume(coriolis_src, v_rhs);

  const auto& weight = ops_.mat_.nodalWeights;
  matrix_vector_view v_lhs(lhs.data());

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        const int rowIndices[3] = {
            idx<n1D>(XH, n, m, l),
            idx<n1D>(YH, n, m, l),
            idx<n1D>(ZH, n, m, l)
        };

        for (int k = 0; k < n1D; ++k) {
          const DoubleType Wnk =  weight(n, k);
          for (int j = 0; j < n1D; ++j) {
            auto WnkWmj = Wnk * weight(m, j);
            for (int i = 0; i < n1D; ++i) {
              const DoubleType fac = WnkWmj * weight(l, i) * vol(k, j, i) * rho(k, j, i);

              const int colIndices[3] = {
                  idx<n1D>(XH, k, j, i),
                  idx<n1D>(YH, k, j, i),
                  idx<n1D>(ZH, k, j, i)
              };

              v_lhs(rowIndices[XH], colIndices[XH]) += 0.0;
              v_lhs(rowIndices[XH], colIndices[YH]) += fac * cor_.Jxy_;
              v_lhs(rowIndices[XH], colIndices[ZH]) += fac * cor_.Jxz_;

              v_lhs(rowIndices[YH], colIndices[XH]) -= fac * cor_.Jxy_;
              v_lhs(rowIndices[YH], colIndices[YH]) += 0.0;
              v_lhs(rowIndices[YH], colIndices[ZH]) += fac * cor_.Jyz_;

              v_lhs(rowIndices[ZH], colIndices[XH]) -= fac * cor_.Jxz_;
              v_lhs(rowIndices[ZH], colIndices[YH]) -= fac * cor_.Jxz_;
              v_lhs(rowIndices[ZH], colIndices[ZH]) += 0.0;
            }
          }
        }
      }
    }
  }
}

INSTANTIATE_KERNEL_HOSGL(MomentumCoriolisSrcHOElemKernel)

} // namespace nalu
} // namespace Sierra
