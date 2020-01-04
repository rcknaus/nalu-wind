#ifndef STK_SIMD_CONNECTIVITY_MAP_H
#define STK_SIMD_CONNECTIVITY_MAP_H

#include <Kokkos_View.hpp>
#include <memory>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "Kokkos_Core.hpp"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/Types.hpp"
#include "stk_ngp/Ngp.hpp"

namespace stk {
namespace mesh {
struct Entity;
class BulkData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

constexpr stk::mesh::FastMeshIndex invalid_mesh_index =
  stk::mesh::FastMeshIndex{stk::mesh::InvalidOrdinal,
                           stk::mesh::InvalidOrdinal};
KOKKOS_INLINE_FUNCTION bool
valid_mesh_index(stk::mesh::FastMeshIndex index)
{
  return !(
    index.bucket_id == invalid_mesh_index.bucket_id ||
    index.bucket_ord == invalid_mesh_index.bucket_ord);
}

namespace impl {
template <int p>
struct stk_connectivity_map_t
{
  static elem_mesh_index_view<p> invoke(const ngp::Mesh&, stk::mesh::Selector);
};
} // namespace impl
P_INVOKEABLE(stk_connectivity_map)

static constexpr int invalid_offset = -1;
namespace impl {
template <int p>
struct create_offset_map_t
{
  static elem_offset_view<p>
  invoke(const ngp::Mesh&, const stk::mesh::Selector&, ra_entity_row_view_type);
};
} // namespace impl
P_INVOKEABLE(create_offset_map)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
