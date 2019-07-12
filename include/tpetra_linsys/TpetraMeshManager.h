#ifndef TPETRAMESHMANAGER_H_
#define TPETRAMESHMANAGER_H_

#include "FieldTypeDef.h"
#include "MatrixFreeTypes.h"

#include <Tpetra_Map.hpp>
#include <Teuchos_RCP.hpp>

#include <vector>
#include <string>

namespace sierra {
namespace nalu {

class SolutionPointCategorizer;

struct MeshIdData {
  MeshIdData(
    long maxOwnedRowId,
    long maxSharedNotOwnedRowId,
    std::unordered_map<long, long> localIds,
    std::vector<long> ownedGids,
    std::vector<long> sharedNotOwnedGids,
    std::vector<stk::ParallelMachine> sharedPids)
  : maxOwnedRowId_(maxOwnedRowId),
    maxSharedNotOwnedRowId_(maxSharedNotOwnedRowId),
    localIds_(localIds),
    ownedGids_(ownedGids),
    sharedNotOwnedGids_(sharedNotOwnedGids),
    sharedPids_(sharedPids)
  {}

  long maxOwnedRowId_;
  long maxSharedNotOwnedRowId_;
  std::unordered_map<long, long> localIds_;
  std::vector<long> ownedGids_;
  std::vector<long> sharedNotOwnedGids_;
  std::vector<stk::ParallelMachine> sharedPids_;
};

MeshIdData determine_parallel_mesh_id_categorizations(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts = {},
  stk::mesh::PartVector nonconformalParts = {});

Kokkos::View<int*> entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const GlobalIdFieldType& gidField,
  const SolutionPointCategorizer& cat,
  const stk::mesh::Selector& selector,
  const std::unordered_map<long, long>& localIdToGlobalIdMap);

Kokkos::View<int*> entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const GlobalIdFieldType& gidField,
  stk::mesh::PartVector periodicParts);

Kokkos::View<int*> serial_entity_offset_to_row_lid_map(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector);

Teuchos::RCP<map_type> operator_map(
  const stk::mesh::BulkData& bulk,
  MeshIdData meshData);

Teuchos::RCP<map_type>
operator_map(const stk::mesh::BulkData& bulk,
  GlobalIdFieldType& gidField,
  stk::mesh::Selector activeSelector,
  stk::mesh::PartVector periodicParts,
  stk::mesh::PartVector nonconformalParts);

}
}

#endif /* TPETRAMESHMANAGER_H_ */
