
#include "MomentumDiagonal.h"

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/MomentumKernel.h"
#include "TpetraLinearSystem.h"

namespace sierra { namespace nalu {


template <int p>
MomentumInteriorDiagonal<p>::MomentumInteriorDiagonal(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  TpetraLinearSystem& linsys,
  const VectorFieldType& coordField)
: linsys_(linsys)
{
  initialize(bulk, selector, coordField);
}

template <int p> void MomentumInteriorDiagonal<p>::initialize(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  const VectorFieldType& coordField)
{
  entities_ = element_entity_view<p>(bulk, selector);
  auto coordview = gather_field<p>(bulk, selector, coordField);

  auto& meta = bulk.mesh_meta_data();
  const auto& rhoField = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")
      ->field_of_state(stk::mesh::StateNP1);
  volume_ = volumes<p>(gather_field<p>(bulk, selector, rhoField), coordview);


  const auto& viscField = meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity")
      ->field_of_state(stk::mesh::StateNone);
  mapped_area_ = mapped_areas<p>(gather_field<p>(bulk, selector, viscField), coordview);
}

template <int p>
void MomentumInteriorDiagonal<p>::set_gamma(double gamma) { gamma_ = gamma; }

template <int p>
void MomentumInteriorDiagonal<p>::set_mdot(ko::scs_scalar_view<p> mdot) { mdot_ = mdot; }

template <int p>
void MomentumInteriorDiagonal<p>::compute_diagonal()
{
  constexpr int n1D = p + 1;
  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    const auto diag = op::lhs_diagonal(index, gamma_, volume_, mapped_area_, mdot_);
    for (int n = 0; n < simdLen && entities_(index, n, 0,0,0).is_local_offset_valid(); ++n){
      for (int k = 0; k < n1D; ++k) {
        for (int j = 0; j < n1D; ++j) {
          for (int i = 0; i < n1D; ++i) {
            linsys_.sumIntoNode(entities_(index,n,k,j,i), stk::simd::get_data(diag(k,j,i),n), 0);
          }
        }
      }
    }
  }
}
template class MomentumInteriorDiagonal<1>;
template class MomentumInteriorDiagonal<2>;
template class MomentumInteriorDiagonal<3>;
template class MomentumInteriorDiagonal<4>;

} // namespace nalu
} // namespace Sierra

