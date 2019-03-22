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

#include <boost/functional/hash.hpp>


#include <set>
#include <limits>
#include <type_traits>

#include <sstream>

#define GID_(gid, ndof, idof)  ((ndof)*((gid)-1)+(idof)+1)
#define LID_(lid, ndof, idof)  ((ndof)*((lid))+(idof))

#define GLOBAL_ENTITY_ID(gid, ndof) ((gid-1)/ndof + 1)
#define GLOBAL_ENTITY_ID_IDOF(gid, ndof) ((gid-1) % ndof)

namespace std {
template<>
struct hash<stk::mesh::EntityVector> {
    size_t operator()(const stk::mesh::EntityVector& entityVec) const
    { return boost::hash_range(entityVec.begin(), entityVec.end()); }
};

}

namespace {

  int64_t flattened_global_offset(int64_t gid, int ndof, int idof) { return ndof * (gid - 1) + idof + 1;  }
//  int64_t flattened_local_offset(int64_t lid, int ndof, int idof) { return ndof * lid + idof + 1; }
//  int64_t flattened_entity_offset(int64_t gid , int ndof) { return (gid-1)/ndof + 1; }
//  int dimension_index(int64_t gid, int ndof) { return (gid - 1) % ndof; }



template <typename BoolOp, typename ExecOp> void
for_each_entity_where_do(const stk::mesh::BucketVector& buckets, BoolOp where, ExecOp exec)
{
  for (const auto* ib : buckets) {
    for (auto e : *ib) {
      if (where(e)) { exec(e); }
    }
  }
}

template <typename CatFunc>
std::vector<int64_t> determine_entity_id_list(
  const sierra::nalu::GlobalIdFieldType& gidField,
  const stk::mesh::BucketVector& activeBuckets,
  CatFunc cat)
{
  size_t numEntities = 0;
  for_each_entity_where_do(activeBuckets, cat, [&numEntities](stk::mesh::Entity) { ++numEntities; });
  std::vector<int64_t> entityIdList; entityIdList.reserve(numEntities);
  for_each_entity_where_do(activeBuckets, cat, [&entityIdList, &gidField](stk::mesh::Entity e) {
    entityIdList.push_back(*stk::mesh::field_data(gidField, e));
  });
  std::sort(entityIdList.begin(), entityIdList.end());
  return entityIdList;
}
}
constexpr int ndof = 1;


namespace sierra {  namespace nalu {

MeshIdData determine_mesh_id_info(stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts)
{

  auto solutionCat = SolutionPointCategorizer(bulk, gidField, periodicParts, nonconformalParts);

  const auto& activeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, activeSelector);
  std::vector<int64_t> ownedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&solutionCat](stk::mesh::Entity e) {
      const SolutionPointStatus status = solutionCat.status(e);
      return !is_skipped(status) && is_owned(status);
  });
  const int64_t maxOwnedRowId = ownedEntityIds.size() * ndof;

  int32_t localId = 0;
  std::vector<int64_t> ownedGids(ndof * ownedEntityIds.size());
  std::unordered_map<int64_t, int64_t> localIds;
  for (unsigned k = 0; k < ownedEntityIds.size(); ++k) {
    const auto id = ownedEntityIds[k];
    localIds[id] = ndof * bulk.identifier(bulk.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < ndof; ++d) {
      ownedGids[ndof*k+d] = flattened_global_offset(id, ndof, d);
    }
  }

  std::vector<int64_t> sharedNotOwnedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&](stk::mesh::Entity e) {
      const SolutionPointStatus status = solutionCat.status(e);
      return (!(is_skipped(status) || is_owned(status)) && is_shared(status));
  });

  int64_t numNodes = 0;
  for_each_entity_where_do(
    activeBuckets,
    [&solutionCat](stk::mesh::Entity e)
    {
      const SolutionPointStatus status = solutionCat.status(e);
      return (!is_skipped(status) && !is_ghosted(status));
    },
    [&numNodes](stk::mesh::Entity) { ++numNodes; }
  );
  const int64_t maxSharedNotOwnedRowId = numNodes * ndof;

  std::vector<int64_t> sharedNotOwnedGids(ndof * sharedNotOwnedEntityIds.size());
  for (unsigned k = 0; k < sharedNotOwnedEntityIds.size(); ++k) {
    const auto id = sharedNotOwnedEntityIds[k];
    localIds[id] = bulk.identifier(bulk.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < ndof; ++d) {
      sharedNotOwnedGids[ndof*k + d] = flattened_global_offset(id, ndof, d);
    }
  }
  return MeshIdData(maxOwnedRowId, maxSharedNotOwnedRowId, localIds, ownedGids, sharedNotOwnedGids, {});
}

int64_t look_up_global_id_for_node(const GlobalIdFieldType& gidField, stk::mesh::Entity e)
{
  return flattened_global_offset(*stk::mesh::field_data(gidField, e), ndof, 0);
}

std::vector<int32_t> entity_offset_to_column_lid_map(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  const stk::mesh::Selector& selector,
  Tpetra::Map<int32_t, int64_t>& totalColsMap)
{
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);

  int32_t nEntities = 0;
  for (const auto* ib: buckets) { nEntities += buckets.size(); }

  auto entityToColLID = std::vector<int32_t>{nEntities, -1};

  for (const auto* ib : buckets) {
    for (auto e : *ib) {
      entityToColLID[e.local_offset()] = totalColsMap.getLocalElement(look_up_global_id_for_node(gidField, e));
    }
  }
  return entityToColLID;
}

stk::mesh::Entity get_entity_from_id(const stk::mesh::BulkData& bulk, int64_t id)
{
  ThrowAssert(bulk.is_valid(bulk.get_entity(stk::topology::NODE_RANK, id)));
  return bulk.get_entity(stk::topology::NODE_RANK, id);
}

std::vector<int32_t> entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  const SolutionPointCategorizer& cat,
  const stk::mesh::Selector& selector,
  const std::unordered_map<int64_t, int64_t>& localIdToGlobalIdMap)
{
  std::vector<int32_t> entityRowMap;
  entityRowMap.reserve(bulk.get_size_of_entity_index_space());
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  for (const auto* ib : buckets) {
    const auto& bucket = *ib;
    for (auto node : *ib) {
      const int64_t globalId = look_up_global_id_for_node(gidField, node);
      auto it = localIdToGlobalIdMap.find(globalId);
      if (it != localIdToGlobalIdMap.end()) {
        const int64_t mappedId = it->second;
        entityRowMap[node.local_offset()] = mappedId;
        if (is_slave(cat.type(node))) {
          const auto masterNode = get_entity_from_id(bulk, globalId);
          entityRowMap[masterNode.local_offset()] = mappedId;
        }
      }
    }
  }
  return entityRowMap;
}

std::unordered_set<stk::mesh::EntityVector> element_to_node_lists(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector& selector)
{
  auto connections = std::unordered_set<stk::mesh::EntityVector>{};
  for (const auto* ib : bulk.get_buckets(stk::topology::ELEM_RANK, selector)) {
    for (auto e : *ib) {
      const auto* nodes  = bulk.begin_nodes(e);
      for (unsigned n = 0; n < bulk.num_nodes(e); ++n) {

      }
    }
  }
  return connections;
}

}}

