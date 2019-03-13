/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TpetraGraphManager_h
#define TpetraGraphManager_h

#include <LinearSystem.h>

#include <KokkosInterface.h>

#include <FieldTypeDef.h>

#include <Kokkos_DefaultNode.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <Tpetra_Experimental_BlockCrsMatrix.hpp>
#include <Tpetra_Experimental_BlockVector.hpp>

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <vector>
#include <string>
#include <unordered_map>

namespace nalu_stk {
class CommNeighbors;
}

namespace sierra {
namespace nalu {

//enum class DOFStatus
//
//class TpetraGraphBuilder
//{
//  TpetraGraphBuilder(stk::mesh::BulkData& bulk, stk::mesh::Selector activeSelector, GlobalIdFieldType& gidField)
//  : bulk_(bulk), activeSelector_(activeSelector), globalIdField_(gidField)
//  {
//    localActiveSelector_ = bulk_.mesh_meta_data().locally_owned_part() & activeSelector_;
//  }
//
//  MeshIdManager populate_mesh_info();
//
//  void build_node_graph(const stk::mesh::PartVector& parts);
//  void build_face_to_node_graph(const stk::mesh::PartVector& parts);
//  void build_edge_to_node_graph(const stk::mesh::PartVector& parts);
//  void build_elem_to_node_graph(const stk::mesh::PartVector& parts);
//  void build_reduced_elem_to_node_graph(const stk::mesh::PartVector& parts);
//  void build_face_elem_to_node_graph(const stk::mesh::PartVector& parts);
//  void build_nonconformal_node_graph(const stk::mesh::PartVector& parts);
//  void build_overset_node_graph(const stk::mesh::PartVector& parts);
//
//  stk::mesh::Selector local_active_union(const stk::mesh::PartVector& parts) { return localActiveSelector_ & stk::mesh::selectUnion(parts); }
//
//private:
//  void build_connected_node_graph(stk::topology::rank_t rank, const stk::mesh::PartVector& parts);
//  void insert_connection(stk::mesh::Entity a, stk::mesh::Entity b);
//  bool entity_is_periodic_correct(stk::mesh::Entity e);
//  stk::mesh::EntityId mesh_global_id(stk::mesh::Entity e) {
//    return static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(globalIdField_, e));
//  }
//
//  stk::mesh::BulkData& bulk_;
//  GlobalIdFieldType& globalIdField_;
//  stk::mesh::Selector activeSelector_;
//  stk::mesh::Selector localActiveSelector_;
//
//  std::unique_ptr<LinSys::Map> totalColsMap_;
//  std::unique_ptr<LinSys::Map> optColsMap_;
//  std::unique_ptr<LinSys::Map> ownedRowsMap_;
//  std::unique_ptr<LinSys::Map> sharedNotOwnedRowsMap_;
//  std::unique_ptr<LinSys::Graph> ownedGraph_;
//  std::unique_ptr<LinSys::Graph> sharedNotOwnedGraph_;
//};

} // namespace nalu
} // namespace Sierra

#endif
