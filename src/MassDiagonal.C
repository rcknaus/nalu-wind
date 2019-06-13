
#include "MassDiagonal.h"

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/MassKernel.h"
#include "TpetraLinearSystem.h"

namespace sierra { namespace nalu {

template <int p> MassInteriorDiagonal<p>::MassInteriorDiagonal(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  TpetraLinearSystem& linsys,
  const VectorFieldType& coordField)
: bulk_(bulk), selector_(selector), linsys_(linsys)
{
  entities_ = element_entity_view<p>(bulk, selector);
  volume_ = volumes<p>(gather_field<p>(bulk, selector, coordField));
}

template <int p> void MassInteriorDiagonal<p>::compute_diagonal()
{
  constexpr int n1D = p + 1;
  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    const auto diag = op::lhs_diagonal(index, volume_);
    for (int n = 0; n < simdLen && entities_(index, n, 0,0,0).is_local_offset_valid(); ++n){
      for (int k = 0; k < n1D; ++k) {
        for (int j = 0; j < n1D; ++j) {
          for (int i = 0; i < n1D; ++i) {
            linsys_.sumIntoNode(entities_(index, n, k, j, i), stk::simd::get_data(diag(k, j, i), n), 0);
          }
        }
      }
    }
  }
}
template class MassInteriorDiagonal<1>;
template class MassInteriorDiagonal<2>;
template class MassInteriorDiagonal<3>;
template class MassInteriorDiagonal<4>;

} // namespace nalu
} // namespace Sierra

