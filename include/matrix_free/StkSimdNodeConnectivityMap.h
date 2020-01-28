#ifndef STK_SIMD_NODE_CONNECTIVITY_MAP_H
#define STK_SIMD_NODE_CONNECTIVITY_MAP_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"

#include <Kokkos_View.hpp>
#include <memory>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "Kokkos_Core.hpp"

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

node_mesh_index_view
simd_node_map(const ngp::Mesh&, const stk::mesh::Selector&);
node_offset_view simd_node_offsets(
  const ngp::Mesh&, const stk::mesh::Selector&, ra_entity_row_view_type);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
