#ifndef SOLUTION_POINT_CATEGORIZER_H
#define SOLUTION_POINT_CATEGORIZER_H

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

enum class SolutionPointStatus
{
  notset = 1 << 0,
  skipped = 1 << 1,
  owned_not_shared = 1 << 2,
  shared_not_owned = 1 << 3,
  owned_and_shared = 1 << 4,
  ghosted = 1 << 5,
  nonconformal= 1 << 6
};
ENABLE_BITMASK_OPERATORS(SolutionPointStatus)

enum class SolutionPointType {
  regular = 1 << 0,
  periodic_master = 1 << 1,
  periodic_slave = 1 << 2,
  nonconformal = 1 << 3
};
ENABLE_BITMASK_OPERATORS(SolutionPointType)


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

  SolutionPointStatus status(stk::mesh::Entity e);
  SolutionPointType type(stk::mesh::Entity e);
private:


  stk::mesh::EntityId nalu_mesh_global_id(stk::mesh::Entity e) {
    return static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(globalIdField_, e));
  }

  stk::mesh::EntityId stk_mesh_global_id(stk::mesh::Entity e) {
    return bulk_.identifier(e);
  }

  SolutionPointStatus regular_status(stk::mesh::Entity e);
  SolutionPointStatus special_status(stk::mesh::Entity e);
  SolutionPointStatus periodic_status(stk::mesh::Entity e);
  SolutionPointStatus nonconformal_status(stk::mesh::Entity e);
  bool is_slave(stk::mesh::Entity e);

  stk::mesh::BulkData& bulk_;
  GlobalIdFieldType& globalIdField_;
  stk::mesh::Selector periodicSelector_;
  stk::mesh::Selector nonconformalSelector_;
};

inline bool is_skipped(SolutionPointStatus status)
{
  return static_cast<bool>(status & SolutionPointStatus::skipped);
}

inline bool is_owned(SolutionPointStatus status)
{
  return static_cast<bool>((status & SolutionPointStatus::owned_not_shared) | (status & SolutionPointStatus::owned_and_shared));
}
inline bool is_shared(SolutionPointStatus status)
{
  return static_cast<bool>((status & SolutionPointStatus::shared_not_owned) | (status & SolutionPointStatus::owned_and_shared));
}
inline bool is_ghosted(SolutionPointStatus status)
{
  return static_cast<bool>(status & SolutionPointStatus::ghosted);
}

}
}

#endif /* SOLUTION_POINT_CATEGORIZER_H */
