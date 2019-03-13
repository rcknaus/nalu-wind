#ifndef TPETRAMESHMANAGER_H_
#define TPETRAMESHMANAGER_H_

#include <tpetra_linsys/EnumClassBitmask.h>
#include <FieldTypeDef.h>

#include <Tpetra_DefaultPlatform.hpp>
#include <Kokkos_DefaultNode.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include "stk_util/environment/Env.hpp"
#include <vector>
#include <string>

namespace sierra {
namespace nalu {

enum class DOFStatus
{
  notset = 1 << 0,
  skipped = 1 << 1,
  owned_not_shared = 1 << 2,
  shared_not_owned = 1 << 3,
  owned_and_shared = 1 << 4,
  ghosted = 1 << 5,
  nonconformal= 1 << 6
};
ENABLE_BITMASK_OPERATORS(DOFStatus);

enum class SpecialNode {
  regular = 1 << 0,
  periodic_master = 1 << 1,
  periodic_slave = 1 << 2,
  nonconformal = 1 << 3
};
ENABLE_BITMASK_OPERATORS(SpecialNode);

class SolutionPointCategorizer
{
public:
  SolutionPointCategorizer(
    stk::mesh::BulkData& bulk,
    GlobalIdFieldType& gidField,
    stk::mesh::PartVector periodicParts,
    stk::mesh::PartVector nonconformalParts)
  : bulk_(bulk),
    globalIdField_(gidField),
    periodicSelector_(stk::mesh::selectUnion(periodicParts)),
    nonconformalSelector_(stk::mesh::selectUnion(nonconformalParts))
  {}

  DOFStatus node_status(stk::mesh::Entity e);
  DOFStatus regular_node_status(stk::mesh::Entity e);
  DOFStatus special_node_status(stk::mesh::Entity e);
  DOFStatus periodic_node_status(stk::mesh::Entity e);
  DOFStatus nonconformal_node_status(stk::mesh::Entity e);
  SpecialNode node_type(stk::mesh::Entity e);
  bool is_slave(stk::mesh::Entity e);

private:
  stk::mesh::EntityId nalu_mesh_global_id(stk::mesh::Entity e) {
    return static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(globalIdField_, e));
  }

  stk::mesh::EntityId stk_mesh_global_id(stk::mesh::Entity e) {
    return bulk_.identifier(e);
  }

  stk::mesh::BulkData& bulk_;
  GlobalIdFieldType& globalIdField_;
  stk::mesh::Selector periodicSelector_;
  stk::mesh::Selector nonconformalSelector_;
};

struct NodeIdInfo {

};

class MeshIdManager
{
public:
  MeshIdManager(
    stk::mesh::BulkData& bulk,
    GlobalIdFieldType& gidField,
    int numDof,
    stk::mesh::Selector activeSelector,
    stk::mesh::PartVector periodicParts,
    stk::mesh::PartVector nonconformalParts);

private:
  stk::mesh::EntityId nalu_mesh_global_id(stk::mesh::Entity e) {
    return static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(globalIdField_, e));
  }

  stk::mesh::EntityId stk_mesh_global_id(stk::mesh::Entity e) {
    return bulk_.identifier(e);
  }

  template <typename CatFunc>
  std::vector<int64_t> determine_entity_id_list(const stk::mesh::BucketVector& buckets, CatFunc cat);


  stk::mesh::BulkData& bulk_;
  GlobalIdFieldType& globalIdField_;
  int numDof_;
  stk::mesh::Selector activeSelector_;
  SolutionPointCategorizer nodeCat_;

  int64_t maxOwnedRowId_;
  int64_t maxSharedNotOwnedRowId_;
  std::unordered_map<int64_t, int32_t> localIds_;
  std::vector<int64_t> ownedGids_;
  std::vector<int64_t> sharedNotOwnedGids_;
};


}
}

#endif /* TPETRAMESHMANAGER_H_ */
