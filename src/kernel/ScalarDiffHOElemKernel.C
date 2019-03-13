/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/ScalarDiffHOElemKernel.h>
#include <kernel/TensorProductCVFEMDiffusion.h>
#include <kernel/TensorProductCVFEMScalarBDF2TimeDerivative.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>

#include <BuildTemplates.h>
#include <FieldTypeDef.h>
#include <MatrixFreeElemSolver.h>
#include <SolutionOptions.h>
#include <ScratchViewsHOMF.h>
#include <TimeIntegrator.h>

#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_topology/topology.hpp>

namespace sierra{
namespace nalu{

template <typename AlgTraits>
ScalarDiffHOElemKernel<AlgTraits>::ScalarDiffHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType *scalarQ,
  ScalarFieldType *diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ),
    diffFluxCoeff_(diffFluxCoeff)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());

  dataPreReqs.add_gathered_nodal_field(*coordinates_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(*scalarQ, 1);
  dataPreReqs.add_gathered_nodal_field(*diffFluxCoeff, 1);
}
//--------------------------------------------------------------------------
template <typename AlgTraits> void
ScalarDiffHOElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViewsHO<DoubleType>& scratchViews)
{
  auto v_coords = scratchViews.get_scratch_view<nodal_vector_view>(*coordinates_);
  auto v_diff = scratchViews.get_scratch_view<nodal_scalar_view>(*diffFluxCoeff_);
  scs_vector_workview work_metric(0);
  auto& metric = work_metric.view();

  auto start_time_diff = std::chrono::steady_clock::now();
  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords, v_diff, metric);
  auto end_time_diff = std::chrono::steady_clock::now();
  timer_diff += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_diff-start_time_diff).count();

  matrix_view v_lhs(lhs.data());
  auto start_time_jac = std::chrono::steady_clock::now();
  tensor_assembly::scalar_diffusion_lhs(ops_, metric, v_lhs);
  auto end_time_jac = std::chrono::steady_clock::now();
  timer_jac += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_jac-start_time_jac).count();

  auto scalar = scratchViews.get_scratch_view<nodal_scalar_view>(*scalarQ_);
  nodal_scalar_view v_rhs(rhs.data());
  auto start_time_resid = std::chrono::steady_clock::now();
  tensor_assembly::scalar_diffusion_rhs(ops_, metric, scalar, v_rhs);
  auto end_time_resid = std::chrono::steady_clock::now();
  timer_resid += std::chrono::duration_cast<std::chrono::duration<double>>(end_time_resid - start_time_resid).count();
}

// add mass
template <typename AlgTraits> void
ScalarDiffHOElemKernel<AlgTraits>::executemf(nodal_scalar_view& rhs, ScratchViewsHOMF<poly_order>& scratchViews)
{
  nodal_vector_view v_coords = scratchViews.get_scratch_view(*coordinates_);
  nodal_scalar_view v_diff = scratchViews.get_scratch_view(*diffFluxCoeff_);

  scs_vector_workview work_metric(0);
  auto& metric = work_metric.view();
  high_order_metrics::compute_diffusion_metric_linear(ops_, v_coords, v_diff, metric);

  nodal_scalar_view scalar = scratchViews.get_scratch_view(*scalarQ_);
  tensor_assembly::scalar_diffusion_rhs(ops_, metric, scalar, rhs);
}

INSTANTIATE_KERNEL_HOSGL(ScalarDiffHOElemKernel)


template <int p>
ScalarConductionHOElemKernel<p>::ScalarConductionHOElemKernel(
  const stk::mesh::BulkData& bulkData,
  SolutionOptions& solnOpts,
  ScalarFieldType& phiField,
  ScalarFieldType& diffusivityField,
  ElemDataRequests& dataPreReqs)
{
  const stk::mesh::MetaData& meta_data = bulkData.mesh_meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, solnOpts.get_coordinates_name());
  dataPreReqs.add_gathered_nodal_field(*coordinates_, 3);

  q_[stk::mesh::StateNP1] = &phiField.field_of_state(stk::mesh::StateNP1);
  dataPreReqs.add_gathered_nodal_field(*q_[stk::mesh::StateNP1], 1);

  q_[stk::mesh::StateN] = &phiField.field_of_state(stk::mesh::StateN);
  dataPreReqs.add_gathered_nodal_field(*q_[stk::mesh::StateN], 1);

  const bool twoState = phiField.number_of_states() == 2;
  q_[stk::mesh::StateNM1] =  twoState ? q_[stk::mesh::StateN] : &phiField.field_of_state(stk::mesh::StateNM1);
  dataPreReqs.add_gathered_nodal_field(*q_[stk::mesh::StateNM1], 1);

  diffusivity_ = &diffusivityField.field_of_state(stk::mesh::StateNone);
  dataPreReqs.add_gathered_nodal_field(*diffusivity_, 1);
}

template <int p> void
ScalarConductionHOElemKernel<p>::setup(const TimeIntegrator& ti)
{
  gamma_[0] = ti.get_gamma1() / ti.get_time_step();
  gamma_[1] = ti.get_gamma2() / ti.get_time_step();
  gamma_[2] = ti.get_gamma3() / ti.get_time_step();
}

template <int p> void
ScalarConductionHOElemKernel<p>::executemf(nodal_scalar_view& rhs, ScratchViewsHOMF<p>& scratchViews)
{
  nodal_vector_view coords = scratchViews.get_scratch_view(*coordinates_);
  nodal_scalar_view phip1 = scratchViews.get_scratch_view(*q_[stk::mesh::StateNP1]);

  {
    nodal_scalar_view diff = scratchViews.get_scratch_view(*diffusivity_);
    scs_vector_workview work_metric(0); auto& metric = work_metric.view();
    high_order_metrics::compute_diffusion_metric_linear(ops_, coords, diff, metric);
    tensor_assembly::scalar_diffusion_rhs(ops_, metric, phip1, rhs);
  }

  {
    nodal_scalar_workview work_vol(0); auto& vol = work_vol.view();
    high_order_metrics::compute_volume_metric_linear(ops_, coords, vol);
    auto phim1 = scratchViews.get_scratch_view(*q_[stk::mesh::StateNM1]);
    auto phip0 = scratchViews.get_scratch_view(*q_[stk::mesh::StateN]);
    auto rhom1 = scratchViews.get_scratch_view(*rho_[stk::mesh::StateNM1]);
    auto rhop0 = scratchViews.get_scratch_view(*rho_[stk::mesh::StateN]);
    auto rhop1 = scratchViews.get_scratch_view(*rho_[stk::mesh::StateNP1]);
    tensor_assembly::scalar_dt_rhs(ops_, vol, gamma_, rhom1, rhop0, rhop1, phim1, phip0, phip1, rhs);
  }
}


} // namespace nalu
} // namespace Sierra
