#ifndef STK_TO_TPETRA_MAP_H
#define STK_TO_TPETRA_MAP_H

#include <memory>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <unordered_map>

#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_ngp/Ngp.hpp"

namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

using map_type = Tpetra::Map<>;

void fill_id_fields(
  const ngp::Mesh&,
  const stk::mesh::Selector&,
  stk::mesh::Field<stk::mesh::EntityId>&,
  stk::mesh::Field<typename map_type::global_ordinal_type>&,
  stk::mesh::PartVector = {});

void fill_tpetra_id_field(
  const ngp::Mesh&,
  const stk::mesh::Selector&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  stk::mesh::Field<typename map_type::global_ordinal_type>&,
  stk::mesh::PartVector = {});

Teuchos::RCP<const map_type> owned_row_map(
  const ngp::Mesh&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  const stk::mesh::Selector&,
  stk::mesh::PartVector = {});

Teuchos::RCP<const map_type> owned_and_shared_row_map(
  const ngp::Mesh&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  const stk::mesh::Field<typename map_type::global_ordinal_type>&,
  const stk::mesh::Selector&,
  stk::mesh::PartVector = {});

std::unordered_map<stk::mesh::EntityId, int> global_to_local_id_map(
  const ngp::Mesh&,
  const stk::mesh::Field<stk::mesh::EntityId>&,
  const stk::mesh::Field<typename map_type::global_ordinal_type>&,
  const stk::mesh::Selector&,
  stk::mesh::PartVector = {});

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
