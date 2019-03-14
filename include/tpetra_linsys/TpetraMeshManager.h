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

struct MeshIdData {
  static constexpr int ndof = 1;
  int64_t maxOwnedRowId_;
  int64_t maxSharedNotOwnedRowId_;
  std::unordered_map<int64_t, int64_t> localIds_;
  std::vector<int64_t> ownedGids_;
  std::vector<int64_t> sharedNotOwnedGids_;
  std::vector<stk::ParallelMachine> sharedPids_;
};

MeshIdData determine_mesh_id_info(stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  int numDof,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts = {},
  stk::mesh::PartVector nonconformalParts = {});

}
}

#endif /* TPETRAMESHMANAGER_H_ */
