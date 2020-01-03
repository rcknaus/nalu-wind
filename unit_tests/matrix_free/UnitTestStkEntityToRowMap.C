#include <mpi.h>

#include <Kokkos_View.hpp>
#include <functional>
#include <iosfwd>
#include <numeric>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <unordered_set>
#include <vector>

#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "gtest/gtest.h"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"

namespace stk {
namespace mesh {
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

class StkEntityToRowMapFixture : public ::testing::Test
{
protected:
  StkEntityToRowMapFixture()
    : meta(3u),
      bulk(meta, MPI_COMM_WORLD, stk::mesh::BulkData::NO_AUTO_AURA),
      gid_field(meta.declare_field<stk::mesh::Field<stk::mesh::EntityId>>(
        stk::topology::NODE_RANK, "global_ids")),
      tpetra_gid_field(
        meta.declare_field<
          stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>>(
          stk::topology::NODE_RANK, "tpetra_global_ids"))
  {
    stk::mesh::put_field_on_mesh(gid_field, meta.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(
      tpetra_gid_field, meta.universal_part(), 1, nullptr);

    stk::io::StkMeshIoBroker io(bulk.parallel());
    const std::string name =
      "generated:1x1x" + std::to_string(bulk.parallel_size());
    io.set_bulk_data(bulk);
    io.add_mesh_database(name, stk::io::READ_MESH);
    io.create_input_mesh();
    io.populate_bulk_data();

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(gid_field, node) = bulk.identifier(node);
      }
    }
    mesh = ngp::Mesh(bulk);
    fill_id_fields(mesh, meta.universal_part(), gid_field, tpetra_gid_field);
  }

  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  stk::mesh::Field<stk::mesh::EntityId>& gid_field;
  stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
    tpetra_gid_field;
  ngp::Mesh mesh;
};

TEST_F(
  StkEntityToRowMapFixture,
  entity_index_map_has_correct_number_of_valid_entries)
{
  auto elid = entity_to_row_lid_mapping(
    mesh, gid_field, tpetra_gid_field,
    meta.locally_owned_part() | meta.globally_shared_part());
  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);

  int valid_count = 0;
  for (int k = 0; k < elid.extent_int(0); ++k) {
    if (elid_h(k) >= 0)
      ++valid_count;
  }
  EXPECT_EQ(valid_count, 8);
}

TEST_F(StkEntityToRowMapFixture, entity_index_map_has_unique_entries)
{
  auto elid = entity_to_row_lid_mapping(
    mesh, gid_field, tpetra_gid_field,
    meta.locally_owned_part() | meta.globally_shared_part());
  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);

  std::set<int> entries;
  for (int k = 0; k < elid.extent_int(0); ++k) {
    entries.emplace(elid_h(k));
  }
  EXPECT_EQ(entries.size(), 8u + 1u);
}
TEST_F(
  StkEntityToRowMapFixture,
  index_entity_map_has_correct_number_of_valid_entries)
{
  auto lide = row_lid_to_mesh_index_mapping(
    mesh, entity_to_row_lid_mapping(
            mesh, gid_field, tpetra_gid_field, meta.universal_part()));
  auto lide_h = Kokkos::create_mirror_view(lide);
  Kokkos::deep_copy(lide_h, lide);

  int valid_count = 0;
  for (int k = 0; k < lide_h.extent_int(0); ++k) {
    auto fmi = lide_h(k);
    if (
      fmi.bucket_id != stk::mesh::InvalidOrdinal &&
      fmi.bucket_ord != stk::mesh::InvalidOrdinal)
      ++valid_count;
  }
  EXPECT_EQ(valid_count, 8);
}

TEST_F(StkEntityToRowMapFixture, index_entity_map_has_unique_entries)
{
  auto lide = row_lid_to_mesh_index_mapping(
    mesh, entity_to_row_lid_mapping(
            mesh, gid_field, tpetra_gid_field, meta.universal_part()));
  auto lide_h = Kokkos::create_mirror_view(lide);
  Kokkos::deep_copy(lide_h, lide);

  struct HashFastMeshIndex
  {
    size_t operator()(const stk::mesh::FastMeshIndex& fmi) const
    {
      return std::hash<unsigned>{}(fmi.bucket_id) ^
             std::hash<unsigned>{}(fmi.bucket_ord);
    }
  };

  struct EquivFastMeshIndex
  {
    bool operator()(
      const stk::mesh::FastMeshIndex& fmi1,
      const stk::mesh::FastMeshIndex& fmi2) const
    {
      return (
        fmi1.bucket_id == fmi2.bucket_id && fmi1.bucket_ord == fmi2.bucket_ord);
    }
  };

  std::unordered_set<
    stk::mesh::FastMeshIndex, HashFastMeshIndex, EquivFastMeshIndex>
    entries;
  for (int k = 0; k < lide_h.extent_int(0); ++k) {
    entries.emplace(lide_h(k));
  }
  EXPECT_EQ(entries.size(), 8u);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
