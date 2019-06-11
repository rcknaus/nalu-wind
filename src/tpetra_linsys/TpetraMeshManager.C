#include <tpetra_linsys/TpetraMeshManager.h>
#include <tpetra_linsys/SolutionPointCategorizer.h>
#include <nalu_make_unique.h>

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <element_promotion/NodeMapMaker.h>

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
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <boost/functional/hash.hpp>


#include <set>
#include <limits>
#include <type_traits>

#include <sstream>

namespace std
{
  template<> struct hash<stk::mesh::EntityVector> {
    size_t operator()(const stk::mesh::EntityVector& entityVec) const
    { return boost::hash_range(entityVec.begin(), entityVec.end()); }
  };
}

namespace sierra { namespace nalu {

namespace {
global_ordinal_type look_up_global_id_for_node(const GlobalIdFieldType& gidField, stk::mesh::Entity e)  {
  return *stk::mesh::field_data(gidField, e);
}

template <typename BoolOp, typename ExecOp>
void for_each_entity_where_do(const stk::mesh::BucketVector& buckets, BoolOp where, ExecOp exec)
{
  for (const auto* ib : buckets) {
    for (auto e : *ib) {
      if (where(e)) {
        exec(e);
      }
    }
  }
}

template <typename CatFunc> std::vector<global_ordinal_type> determine_entity_id_list(
  const GlobalIdFieldType& gidField,
  const stk::mesh::BucketVector& activeBuckets,
  CatFunc cat)
{
  size_t numEntities = 0;
  for_each_entity_where_do(activeBuckets, cat, [&numEntities](stk::mesh::Entity) { ++numEntities; });
  std::vector<global_ordinal_type> entityIdList; entityIdList.reserve(numEntities);
  for_each_entity_where_do(activeBuckets, cat, [&entityIdList, &gidField](stk::mesh::Entity e) {
    entityIdList.push_back(look_up_global_id_for_node(gidField, e));
  });
  std::sort(entityIdList.begin(), entityIdList.end());
  return entityIdList;
}
}

MeshIdData determine_parallel_mesh_id_categorizations(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts)
{
  auto solutionCat = SolutionPointCategorizer(bulk, gidField, periodicParts, nonconformalParts);

  const auto& activeBuckets = bulk.get_buckets(stk::topology::NODE_RANK, activeSelector);
  std::vector<global_ordinal_type> ownedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&solutionCat](stk::mesh::Entity e) {
      const SolutionPointStatus status = solutionCat.status(e);
      return !is_skipped(status) && is_owned(status);
    }
  );
  const global_ordinal_type maxOwnedRowId = ownedEntityIds.size();
  std::unordered_map<global_ordinal_type, global_ordinal_type> localIds;
  for (auto id : ownedEntityIds) {
    localIds[id] = look_up_global_id_for_node(gidField, bulk.get_entity(stk::topology::NODE_RANK, id));
  }

  std::vector<global_ordinal_type> sharedNotOwnedEntityIds = determine_entity_id_list(
    gidField,
    activeBuckets,
    [&solutionCat](stk::mesh::Entity e)
    {
      const SolutionPointStatus status = solutionCat.status(e);
      return (!(is_skipped(status) || is_owned(status)) && is_shared(status));
    }
  );

  std::vector<global_ordinal_type> sharedNotOwnedGids(sharedNotOwnedEntityIds.size());
  for (auto id : sharedNotOwnedEntityIds) {
    localIds[id] = look_up_global_id_for_node(gidField, bulk.get_entity(stk::topology::NODE_RANK, id));
  }

  global_ordinal_type numNodes = 0;
  for_each_entity_where_do(
    activeBuckets,
    [&solutionCat](stk::mesh::Entity e)
    {
      const SolutionPointStatus status = solutionCat.status(e);
      return (!is_skipped(status) && !is_ghosted(status));
    },
    [&numNodes](stk::mesh::Entity) { ++numNodes; }
  );
  const global_ordinal_type maxSharedNotOwnedRowId = numNodes;

  return MeshIdData(maxOwnedRowId, maxSharedNotOwnedRowId, localIds, ownedEntityIds, sharedNotOwnedEntityIds, {});
}


stk::mesh::Entity get_entity_from_id(const stk::mesh::BulkData& bulk, global_ordinal_type id)
{
  ThrowAssert(bulk.is_valid(bulk.get_entity(stk::topology::NODE_RANK, id)));
  return bulk.get_entity(stk::topology::NODE_RANK, id);
}

Kokkos::View<local_ordinal_type*> entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  const SolutionPointCategorizer& cat,
  const stk::mesh::Selector& selector,
  const std::unordered_map<global_ordinal_type, global_ordinal_type>& localIdToGlobalIdMap)
{
  Kokkos::View<local_ordinal_type*> entityRowMap("entity_row_map", bulk.get_size_of_entity_index_space());
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  for (const auto* ib : buckets) {
    const auto& bucket = *ib;
    for (auto node : bucket) {
      const global_ordinal_type globalId = bulk.identifier(node)-1;
      auto it = localIdToGlobalIdMap.find(globalId);
      if (it != localIdToGlobalIdMap.end()) {
        const global_ordinal_type mappedId = it->second - 1;
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

Kokkos::View<local_ordinal_type*> entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const GlobalIdFieldType& gidField,
  stk::mesh::PartVector periodicParts)
{
  auto categorizer = SolutionPointCategorizer(bulk, gidField, periodicParts, {});
  auto data = determine_parallel_mesh_id_categorizations(bulk, gidField, selector);
  return entity_offset_to_row_lid_map(bulk, gidField, categorizer, selector, data.localIds_);
}

Kokkos::View<local_ordinal_type*> serial_entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
{
  Kokkos::View<local_ordinal_type*> entityRowMap("entity_row_map",  bulk.get_size_of_entity_index_space());
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  for (const auto* ib : buckets) {
    for (auto node : *ib) {
      entityRowMap(node.local_offset()) = bulk.identifier(node) - 1;
    }
  }
  return entityRowMap;
}

Teuchos::RCP<Tpetra::Map<local_ordinal_type, global_ordinal_type>>
operator_map(const stk::mesh::BulkData& bulk, MeshIdData meshData)
{
  size_t numLocalElems = meshData.ownedGids_.size();
  size_t numGlobalElems = 0;
  stk::all_reduce_sum(bulk.parallel(), &numLocalElems, &numGlobalElems, 1);

  return make_rcp<Tpetra::Map<local_ordinal_type, global_ordinal_type>>(
    numGlobalElems,
    0,
    make_rcp<Teuchos::MpiComm<int>>(bulk.parallel())
  );
}

Teuchos::RCP<Tpetra::Map<local_ordinal_type, global_ordinal_type>>
operator_map(const stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  stk::mesh::Selector active,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts)
{
  return operator_map(bulk, determine_parallel_mesh_id_categorizations(bulk, gidField, active, periodicParts, nonconformalParts));
}

}} //sierra::nalu

