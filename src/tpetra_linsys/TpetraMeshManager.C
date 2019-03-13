#include <tpetra_linsys/TpetraMeshManager.h>
#include <nalu_make_unique.h>

#include <stk_util/util/SortAndUnique.hpp>

#include <NaluCommNeighbors.hpp>
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

namespace sierra {
namespace nalu {

// this stuff probably should be handled through part membership relations

DOFStatus SolutionPointCategorizer::regular_node_status(stk::mesh::Entity node)
{
  const stk::mesh::Bucket& b = bulk_.bucket(node);
  const bool entityIsOwned = b.owned();
  const bool entityIsShared = b.shared();

  if (entityIsOwned && entityIsShared) {
    return DOFStatus::owned_and_shared;
  }
  else if (!entityIsOwned && entityIsShared) {
    return DOFStatus::shared_not_owned;
  }
  else if (entityIsOwned && !entityIsShared) {
    return DOFStatus::owned_not_shared;
  }
  else if (!entityIsOwned && !entityIsShared) {
    return DOFStatus::ghosted;
  }
  return DOFStatus::skipped;
}

DOFStatus SolutionPointCategorizer::nonconformal_node_status(stk::mesh::Entity node)
{
  return regular_node_status(node) | DOFStatus::nonconformal;
}

bool SolutionPointCategorizer::is_slave(stk::mesh::Entity node)
{
  // this rule could potentially be invalidated in the future
  return (stk_mesh_global_id(node) != nalu_mesh_global_id(node));
}

DOFStatus SolutionPointCategorizer::periodic_node_status(stk::mesh::Entity node)
{
  const bool isSlaveNode = is_slave(node);
  if (!isSlaveNode) {
    return regular_node_status(node);
  }
  else {
    stk::mesh::Entity masterEntity = bulk_.get_entity(stk::topology::NODE_RANK, nalu_mesh_global_id(node));
    if (bulk_.is_valid(masterEntity)) {
      return (DOFStatus::skipped | node_status(masterEntity));
    }
  }
  return DOFStatus::skipped;
}

SpecialNode SolutionPointCategorizer::node_type(stk::mesh::Entity node)
{
  const stk::mesh::Bucket& b = bulk_.bucket(node);
  bool nodeInteractsPeriodic = false;
  bool nodeInteractsNonConf = false;
  for (auto part : b.supersets()) {
    if (periodicSelector_(*part)) {
      nodeInteractsPeriodic = true;
    }
    if (nonconformalSelector_(*part)) {
      nodeInteractsNonConf = true;
    }
  }

  if (nodeInteractsPeriodic && nodeInteractsNonConf) {
    if (is_slave(node)) {
      return SpecialNode::periodic_slave | SpecialNode::nonconformal;
    }
    return SpecialNode::periodic_master | SpecialNode::nonconformal;
  }
  else if (nodeInteractsPeriodic && !nodeInteractsNonConf) {
    if (is_slave(node)) {
      return SpecialNode::periodic_slave;
    }
    return SpecialNode::periodic_master;
  }
  else if (!nodeInteractsPeriodic && nodeInteractsNonConf) {
    return SpecialNode::nonconformal;
  }
  return SpecialNode::regular;
}

DOFStatus SolutionPointCategorizer::node_status(stk::mesh::Entity e)
{
   auto nodeType = node_type(e);
   ThrowRequireMsg(nodeType != (SpecialNode::periodic_master | SpecialNode::nonconformal) &&
     nodeType != (SpecialNode::periodic_slave | SpecialNode::nonconformal),"Node id:"
     + std::to_string(bulk_.identifier(e)) + " is both periodic and nonconformal");

   switch (nodeType) {
     case SpecialNode::regular: {
       return regular_node_status(e);
     }
     case SpecialNode::periodic_master: {
       return periodic_node_status(e);
     }
     case SpecialNode::periodic_slave: {
       return periodic_node_status(e);
     }
     case SpecialNode::nonconformal:
       return nonconformal_node_status(e);
     default: {
       return DOFStatus::skipped;
     }
   }
}

namespace
{

bool is_skipped(DOFStatus status) { return static_cast<bool>(status & DOFStatus::skipped); }
bool is_owned(DOFStatus status) { return static_cast<bool>((status & DOFStatus::owned_not_shared) | (status & DOFStatus::owned_and_shared)); }
bool is_shared(DOFStatus status) { return static_cast<bool>((status & DOFStatus::shared_not_owned) | (status & DOFStatus::owned_and_shared)); }
bool is_ghosted(DOFStatus status) { return static_cast<bool>(status & DOFStatus::ghosted);}

template <typename BoolFunc, typename OpFunc> void
for_each_entity_where_do(const stk::mesh::BucketVector& buckets, BoolFunc where, OpFunc op)
{
  for (const auto* ib : buckets) {
    for (size_t k = 0u; k < ib->size(); ++k) {
      const auto e = (*ib)[k];
      if (where(e)) { op(e); }
    }
  }
}

}

MeshIdManager::MeshIdManager(
  stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  int numDof,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts)
: bulk_(bulk),
  globalIdField_(gidField),
  numDof_(numDof),
  activeSelector_(activeSelector),
  nodeCat_(bulk, gidField, periodicParts, nonconformalParts)
{
  const auto& activeBuckets = bulk_.get_buckets(stk::topology::NODE_RANK, activeSelector_);
  std::vector<int64_t> ownedEntityIds = determine_entity_id_list(activeBuckets,
    [&](stk::mesh::Entity e) {
    const DOFStatus status = nodeCat_.node_status(e);
    return (!is_skipped(status) && is_owned(status));
  });
  maxOwnedRowId_ = ownedEntityIds.size() * numDof_;

  ownedGids_.reserve(numDof_ * ownedEntityIds.size());
  for (const auto id : ownedEntityIds) {
    localIds_[id] = bulk_.identifier(bulk_.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < numDof_; ++d) {
      ownedGids_.push_back(GID_(id, numDof_, d));
    }
  }

  std::vector<int64_t> sharedNotOwnedEntityIds = determine_entity_id_list(activeBuckets,
    [&](stk::mesh::Entity e) {
      const DOFStatus status = nodeCat_.node_status(e);
      return (!is_skipped(status) && !is_owned(status) && is_shared(status));
  });

  int64_t numNodes = 0;
  for_each_entity_where_do(activeBuckets,
    [&](stk::mesh::Entity e) {
    const DOFStatus status = nodeCat_.node_status(e);
    return (!is_skipped(status) && !is_ghosted(status));
  },
  [&](stk::mesh::Entity e) {
    ++numNodes;
  });
  maxSharedNotOwnedRowId_ = numNodes * numDof_;

  sharedNotOwnedGids_.reserve(numDof_ *  sharedNotOwnedEntityIds .size());
  for (const auto id : sharedNotOwnedEntityIds) {
    localIds_[id] = bulk_.identifier(bulk_.get_entity(stk::topology::NODE_RANK, id));
    for (int d = 0; d < numDof_; ++d) {
      sharedNotOwnedGids_.push_back(GID_(id, numDof_, d));
    }
  }
}


template <typename CatFunc>
std::vector<int64_t> MeshIdManager::determine_entity_id_list(const stk::mesh::BucketVector& activeBuckets, CatFunc cat)
{
  size_t numEntities = 0;
  for_each_entity_where_do(activeBuckets, cat, [&](stk::mesh::Entity e) { ++numEntities; });

  std::vector<int64_t> entityIdList; entityIdList.reserve(numEntities);
  for_each_entity_where_do(activeBuckets, cat, [&](stk::mesh::Entity e) {
    entityIdList.push_back(nalu_mesh_global_id(e));
  });

  std::sort(entityIdList.begin(), entityIdList.end());
  return entityIdList;
}


}
}
