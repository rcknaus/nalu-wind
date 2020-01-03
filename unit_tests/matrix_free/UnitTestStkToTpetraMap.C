#include "matrix_free/StkToTpetraMap.h"
#include "StkConductionFixture.h"

#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_Map_decl.hpp>
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
#include <vector>

#include "gtest/gtest.h"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_ngp/Ngp.hpp"

namespace stk {
namespace mesh {
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

class StkToTpetraMapFixture : public ::testing::Test
{
protected:
  StkToTpetraMapFixture()
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

TEST_F(StkToTpetraMapFixture, successful_owned_map_creation)
{
  EXPECT_GT(
    owned_row_map(mesh, gid_field, meta.universal_part())
      .get()
      ->getMaxGlobalIndex(),
    1);
}

TEST_F(StkToTpetraMapFixture, owned_map_has_correct_local_size)
{
  // this test relies on STKs node sharing policy of lowest rank winning.  If
  // that ever changes, this will fail
  if (bulk.parallel_size() > 2)
    return;
  const unsigned expected_local_size =
    (bulk.parallel_rank() % 2 == 0) ? 8u : 4u;
  EXPECT_EQ(
    owned_row_map(mesh, gid_field, meta.universal_part())->getNodeNumElements(),
    expected_local_size);
}

TEST_F(StkToTpetraMapFixture, owned_map_has_correct_global_size)
{
  const size_t expected_global_size = 8 + (bulk.parallel_size() - 1) * 4;
  EXPECT_EQ(
    owned_row_map(mesh, gid_field, meta.universal_part())
      ->getGlobalNumElements(),
    expected_global_size);
}

TEST_F(StkToTpetraMapFixture, successful_owned_and_shared_map_creation)
{
  const auto shared_map = owned_and_shared_row_map(
    mesh, gid_field, tpetra_gid_field, meta.universal_part());
  EXPECT_TRUE(shared_map.get() != nullptr);
}

TEST_F(StkToTpetraMapFixture, correct_owned_and_shared_local_index_sum)
{
  const auto oas_map = owned_and_shared_row_map(
    mesh, gid_field, tpetra_gid_field, meta.universal_part());
  EXPECT_EQ(oas_map->getNodeNumElements(), 8u);
}

TEST_F(StkToTpetraMapFixture, owned_and_shared_is_just_owned_in_serial)
{
  if (bulk.parallel_size() != 1) {
    return;
  }
  const auto owned_map = owned_row_map(mesh, gid_field, meta.universal_part());
  const auto oas_map = owned_and_shared_row_map(
    mesh, gid_field, tpetra_gid_field, meta.universal_part());
  EXPECT_EQ(oas_map->getNodeNumElements(), owned_map->getNodeNumElements());
}
TEST_F(StkToTpetraMapFixture, owned_and_shared_globally_is_whole_shattered_mesh)
{
  const size_t expected_global_size = bulk.parallel_size() * 8;
  const auto oas_map = owned_and_shared_row_map(
    mesh, gid_field, tpetra_gid_field, meta.universal_part());
  EXPECT_EQ(oas_map->getGlobalNumElements(), expected_global_size);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
