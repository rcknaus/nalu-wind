#include <tpetra_linsys/SolutionPointCategorizer.h>
#include <nalu_make_unique.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

#include <Kokkos_Serial.hpp>


namespace sierra {
namespace nalu {

// this stuff probably should be handled through part membership relations

SolutionPointStatus SolutionPointCategorizer::regular_status(stk::mesh::Entity node)
{
  const stk::mesh::Bucket& b = bulk_.bucket(node);
  const bool entityIsOwned = b.owned();
  const bool entityIsShared = b.shared();

  if (entityIsOwned && entityIsShared) {
    return SolutionPointStatus::owned_and_shared;
  }
  else if (!entityIsOwned && entityIsShared) {
    return SolutionPointStatus::shared_not_owned;
  }
  else if (entityIsOwned && !entityIsShared) {
    return SolutionPointStatus::owned_not_shared;
  }
  else if (!entityIsOwned && !entityIsShared) {
    return SolutionPointStatus::ghosted;
  }
  return SolutionPointStatus::skipped;
}

SolutionPointStatus SolutionPointCategorizer::nonconformal_status(stk::mesh::Entity node)
{
  return regular_status(node) | SolutionPointStatus::nonconformal;
}

bool SolutionPointCategorizer::is_slave(stk::mesh::Entity node)
{
  // this rule could potentially be invalidated in the future
  return (stk_mesh_global_id(node) != nalu_mesh_global_id(node));
}

SolutionPointStatus SolutionPointCategorizer::periodic_status(stk::mesh::Entity node)
{
  if (!is_slave(node)) {
    return regular_status(node);
  }
  else {
    stk::mesh::Entity masterEntity = bulk_.get_entity(stk::topology::NODE_RANK, nalu_mesh_global_id(node));
    if (bulk_.is_valid(masterEntity)) {
      return (SolutionPointStatus::skipped | status(masterEntity));
    }
  }
  return SolutionPointStatus::skipped;
}

SolutionPointCategorizer::SolutionPointType SolutionPointCategorizer::type(stk::mesh::Entity solPoint)
{
  const stk::mesh::Bucket& b = bulk_.bucket(solPoint);
  bool periodicSolPoint = false;
  bool nonconformalSolPoint = false;
  for (auto part : b.supersets()) {
    if (periodicSelector_(*part)) {
      periodicSolPoint = true;
    }
    if (nonconformalSelector_(*part)) {
      nonconformalSolPoint = true;
    }
  }

  if (periodicSolPoint && nonconformalSolPoint) {
    if (is_slave(solPoint)) {
      return SolutionPointType::periodic_slave | SolutionPointType::nonconformal;
    }
    return SolutionPointType::periodic_master | SolutionPointType::nonconformal;
  }
  else if (periodicSolPoint && !nonconformalSolPoint) {
    if (is_slave(solPoint)) {
      return SolutionPointType::periodic_slave;
    }
    return SolutionPointType::periodic_master;
  }
  else if (!periodicSolPoint && nonconformalSolPoint) {
    return SolutionPointType::nonconformal;
  }
  return SolutionPointType::regular;
}

SolutionPointStatus SolutionPointCategorizer::status(stk::mesh::Entity e)
{
  ThrowAssertMsg(type(e) != (SolutionPointType::periodic_master | SolutionPointType::nonconformal) &&
    type(e) != (SolutionPointType::periodic_slave | SolutionPointType::nonconformal),"Node id:"
    + std::to_string(bulk_.identifier(e)) + " is both periodic and nonconformal");

  switch (type(e)) {
    case SolutionPointType::regular: {
      return regular_status(e);
    }
    case SolutionPointType::periodic_master: {
      return periodic_status(e);
    }
    case SolutionPointType::periodic_slave: {
      return periodic_status(e);
    }
    case SolutionPointType::nonconformal:
      return nonconformal_status(e);
    default: {
      return SolutionPointStatus::skipped;
    }
  }
}

}
}
