#include <tpetra_linsys/TpetraMeshManager.h>
#include <tpetra_linsys/SolutionPointCategorizer.h>
#include <nalu_make_unique.h>

#include <stk_util/util/SortAndUnique.hpp>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/environment/WallTime.hpp>
#include <stk_util/util/SortAndUnique.hpp>

#include <stk_util/parallel/ParallelReduce.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <Kokkos_Serial.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Details_shortSort.hpp>
#include <Tpetra_Details_makeOptimizedColMap.hpp>

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <set>
#include <limits>
#include <type_traits>

#include <sstream>

#define GID_(gid, ndof, idof)  ((ndof)*((gid)-1)+(idof)+1)
#define LID_(lid, ndof, idof)  ((ndof)*((lid))+(idof))

#define GLOBAL_ENTITY_ID(gid, ndof) ((gid-1)/ndof + 1)
#define GLOBAL_ENTITY_ID_IDOF(gid, ndof) ((gid-1) % ndof)

namespace {

template <typename BoolOp, typename ExecOp> void
for_each_entity_where_do(const stk::mesh::BucketVector& buckets, BoolOp where, ExecOp exec)
{
  for (const auto* ib : buckets) {
    for (auto e : *ib) {
      if (where(e)) { exec(e); }
    }
  }\
}

template <typename CatFunc>
std::vector<int64_t> determine_entity_id_list(
  const sierra::nalu::GlobalIdFieldType& gidField,
  const stk::mesh::BucketVector& activeBuckets,
  CatFunc cat) // cats are doing funcs now?!!?! 2019
{
  size_t numEntities = 0;
  for_each_entity_where_do(activeBuckets, cat, [&numEntities](stk::mesh::Entity) { ++numEntities; });

  std::vector<int64_t> entityIdList; entityIdList.reserve(numEntities);
  for_each_entity_where_do(activeBuckets, cat, [&entityIdList, &gidField](stk::mesh::Entity e) {
    entityIdList.push_back(static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(gidField, e)));
  });

  std::sort(entityIdList.begin(), entityIdList.end());
  return entityIdList;
}

}

namespace sierra {  namespace nalu {

MeshIdData determine_mesh_id_info(stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts)
{
  auto cat = SolutionPointCategorizer(bulk, gidField, periodicParts, nonconformalParts);

  const auto& activeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, activeSelector);
  std::vector<int64_t> ownedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&cat](stk::mesh::Entity e) {
    const SolutionPointStatus status = cat.status(e);
    return (!is_skipped(status) && is_owned(status));
  });
  const int64_t maxOwnedRowId = ownedEntityIds.size() * MeshIdData::ndof;

  std::vector<int64_t> ownedGids(MeshIdData::ndof * ownedEntityIds.size());
  std::unordered_map<int64_t, int64_t> localIds;
  for (const auto id : ownedEntityIds) {
    localIds[id] = bulk.identifier(bulk.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < MeshIdData::ndof; ++d) {
      ownedGids.push_back(GID_(id, MeshIdData::ndof, d));
    }
  }

  std::vector<int64_t> sharedNotOwnedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&](stk::mesh::Entity e) {
      const SolutionPointStatus status = cat.status(e);
      return (!is_skipped(status) && !is_owned(status) && is_shared(status));
  });

  int64_t numNodes = 0;
  for_each_entity_where_do(
    activeBuckets,
    [&cat](stk::mesh::Entity e)
    {
      const SolutionPointStatus status = cat.status(e);
      return (!is_skipped(status) && !is_ghosted(status));
    },
    [&numNodes](stk::mesh::Entity) { ++numNodes; }
  );
  const int64_t maxSharedNotOwnedRowId = numNodes * MeshIdData::ndof;

  std::vector<int64_t> sharedNotOwnedGids(MeshIdData::ndof, sharedNotOwnedEntityIds.size());
  for (const auto id : sharedNotOwnedEntityIds) {
    localIds[id] = bulk.identifier(bulk.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < MeshIdData::ndof; ++d) {
      sharedNotOwnedGids.push_back(GID_(id, MeshIdData::ndof, d));
    }
  }

  return MeshIdData { maxOwnedRowId, maxSharedNotOwnedRowId, localIds, ownedGids, sharedNotOwnedGids };
}

std::vector<std::vector<stk::mesh::Entity>> element_to_node_map(const stk::mesh::BulkData& bulk, stk::mesh::Selector& selector)
{
  for (const auto* ib : bulk.get_buckets(stk::topology::ELEM_RANK, selector)) {
    for (auto e : *ib) {
      const auto* nodes  = bulk.begin_nodes(e);
    }
  }

}

}}

