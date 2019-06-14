
#include "ConductionDiagonal.h"

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"
#include "MatrixFreeTraits.h"


#include "kernel/ConductionKernel.h"
#include "TpetraLinearSystem.h"

namespace sierra { namespace nalu {


template <int p>
ConductionInteriorDiagonal<p>::ConductionInteriorDiagonal(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  TpetraLinearSystem& linsys,
  const VectorFieldType& coordField,
  const ScalarFieldType& alpha,
  const ScalarFieldType& diffusivity)
: bulk_(bulk), selector_(selector), linsys_(linsys)
{
  entities_ = element_entity_view<p>(bulk, selector);
  auto coordview = gather_field<p>(bulk, selector, coordField);
  volume_ = volumes<p>(gather_field<p>(bulk, selector, alpha), coordview);
  mapped_area_ = mapped_areas<p>(gather_field<p>(bulk, selector, diffusivity), coordview);
}

template <int p>
void ConductionInteriorDiagonal<p>::initialize_connectivity()
{
  linsys_.buildNodeGraph(selector_);
}

template <int p>
void ConductionInteriorDiagonal<p>::set_gamma(double gamma) { gamma_ = gamma; }

template <int p>
void ConductionInteriorDiagonal<p>::compute_diagonal()
{
  constexpr int n1D = p + 1;
  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    const auto diag = op::lhs_diagonal(index, gamma_, volume_, mapped_area_);
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
template class ConductionInteriorDiagonal<POLY1>;
template class ConductionInteriorDiagonal<POLY2>;
template class ConductionInteriorDiagonal<POLY3>;
template class ConductionInteriorDiagonal<POLY4>;

} // namespace nalu
} // namespace Sierra

