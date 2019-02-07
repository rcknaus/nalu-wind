/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <NaluEnv.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>
#include <NaluParsingHelper.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/GetBuckets.hpp>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/util/SortAndUnique.hpp>

#include <xfer/Transfer.h>
#include <xfer/Transfers.h>
#include <xfer/FromMesh.h>
#include <xfer/ToMesh.h>
#include <xfer/LinInterp.h>
#include <stk_transfer/GeometricTransfer.hpp>

#include <stk_mesh/base/GetEntities.hpp>

#include <stk_mesh/base/HashEntityAndEntityKey.hpp>
// stk_search
#include <stk_search/SearchMethod.hpp>

#include "boost/make_shared.hpp"

namespace sierra{
namespace nalu{

  const VectorFieldType& coord_field(const stk::mesh::BulkData& bulk)
  {
    return *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
  }

  std::pair<std::array<double,3>, double> parametric_coords_for_point(
    MasterElement& me,
    const stk::mesh::Entity* elem_nodes,
    std::vector<double>& elemCoordsScratch,
    const VectorFieldType& coordField,
    const double* const pointLocation)
  {
    const int nNodes = me.nodesPerElement_;
    const int dim = me.nDim_;

    elemCoordsScratch.resize(nNodes*dim);
    for (int n = 0; n < nNodes; ++n) {
      const double* const coordData = stk::mesh::field_data(coordField, elem_nodes[n]);
      for (int d = 0; d < dim; ++d) {
        elemCoordsScratch[nNodes * d + n] = coordData[d];
      }
    }
    std::array<double,3> parametricCoords{{}};
    const double parametricDistance = me.isInElement(elemCoordsScratch.data(), pointLocation, parametricCoords.data());
    return {parametricCoords, parametricDistance};
  }

  void interpolate_point(
    MasterElement& me,
    const stk::mesh::Entity* elem_nodes,
    std::vector<double>& elemFieldScratch,
    const stk::mesh::FieldBase& field,
    const double* const parametricCoords,
    double* values)
  {
    const int nNodes = me.nodesPerElement_;
    const int fieldSize = field.max_size(stk::topology::NODE_RANK);
    elemFieldScratch.resize(nNodes * fieldSize);
    for (int n = 0; n < nNodes; ++n) {
      const double* const coordData = static_cast<const double* const>(stk::mesh::field_data(field, elem_nodes[n]));
      for (int d = 0; d < fieldSize; ++d) {
        elemFieldScratch[nNodes * d + n] = coordData[d];
      }
    }
    me.interpolatePoint(fieldSize, parametricCoords, elemFieldScratch.data(), values);
  }

  std::vector<std::pair<stk::search::Sphere<double>, stk::search::IdentProc<stk::mesh::EntityKey>>> generate_bounding_spheres_for_nodes(
    stk::mesh::BulkData& bulk,
    stk::mesh::EntityVector nodeList,
    double searchRadius)
  {
    std::vector<std::pair<stk::search::Sphere<double>, stk::search::IdentProc<stk::mesh::EntityKey>>> v;
    v.reserve(nodeList.size());

    auto& coordField = coord_field(bulk);
    const int coordSize = bulk.mesh_meta_data().spatial_dimension();
    for (auto node : nodeList) {
      const double* coords = stk::mesh::field_data(coordField, node);
      const stk::search::Point<double> pt = (coordSize == 2) ?
            stk::search::Point<double>(coords[0], coords[1]) :
            stk::search::Point<double>(coords[0], coords[1], coords[2]);
      v.emplace_back(
        stk::search::Sphere<double>{pt, searchRadius},
        stk::search::IdentProc<stk::mesh::EntityKey>{bulk.entity_key(node), bulk.parallel_rank()}
      );
    }
    return v;
  }

  std::vector<std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>>
  generate_bounding_boxes_for_elements(stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
  {

    const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);
    const int elemCount = stk::mesh::count_selected_entities(selector, elem_buckets);
    std::vector<std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>> v;
    v.reserve(elemCount);

    const auto& coordField = coord_field(bulk);
    const int dim = bulk.mesh_meta_data().spatial_dimension();

    for (const auto* ib : elem_buckets) {
      for (auto elem : *ib) {

        const auto minDouble = std::numeric_limits<double>::lowest();
        const auto maxDouble = std::numeric_limits<double>::max();

        stk::search::Point<double> minCorner = (dim == 2) ?
              stk::search::Point<double>(maxDouble, maxDouble) :
              stk::search::Point<double>(maxDouble, maxDouble, maxDouble);

        stk::search::Point<double> maxCorner = (dim == 2) ?
              stk::search::Point<double>(minDouble, minDouble) :
              stk::search::Point<double>(minDouble, minDouble, minDouble);

        const int nNodes = bulk.num_nodes(elem);
        const auto* elemNodes = bulk.begin_nodes(elem);
        for (int n = 0; n < nNodes; ++n) {
          const double* const coordData = stk::mesh::field_data(coordField, elemNodes[n]);
          for (int d = 0; d < dim; ++d) {
            minCorner[d] = std::min(minCorner[d], coordData[d]);
            maxCorner[d] = std::max(maxCorner[d], coordData[d]);
          }
        }

        v.emplace_back(
          stk::search::Box<double>{minCorner, maxCorner},
          stk::search::IdentProc<stk::mesh::EntityKey>{bulk.entity_key(elem), bulk.parallel_rank()}
        );

      }
    }
    std::sort(v.begin(), v.end(), [](
        const std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>& a,
        const std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>& b)
        {  return a.second.id() < b.second.id(); }
    );
    return v;
  }

  template <typename T>
  bool is_empty_on_all_procs(stk::ParallelMachine comm, const T& vec) {
    const int isEmptyLocal = vec.empty() ? 1 : 0;
    int isEmptyGlobal = 0;
    stk::all_reduce_sum(comm, &isEmptyLocal, &isEmptyGlobal, 1);
    return (isEmptyGlobal == 0);
  }

  class NodeMesh
  {
  public:
    using EntityKey = stk::mesh::EntityKey;
    using EntityProc = stk::search::IdentProc<stk::mesh::EntityKey>;
    using EntityProcVec = std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>;
    using BoundingBox = std::pair<stk::search::Sphere<double>, stk::search::IdentProc<stk::mesh::EntityKey>>;

    NodeMesh(stk::mesh::BulkData& bulk, stk::mesh::EntityVector nodeList)
    : bulk_(bulk), nodeList_(nodeList)
    {}

    stk::ParallelMachine comm() const { return bulk_.parallel(); }

    void set_ghosting(stk::mesh::Ghosting* ghosting) { ghosting_ = ghosting; }

    void update_values()
    {
//      if (ghosting_ != nullptr) {
//        stk::mesh::communicate_field_data(*ghosting_, fields_);
//        stk::mesh::copy_owned_to_shared(bulk_, fields_);
//      }
    }

    void bounding_boxes(std::vector<BoundingBox>& v) const
    {
      v = generate_bounding_spheres_for_nodes(bulk_, nodeList_, 1.0e-3);
    }

    stk::mesh::BulkData& bulk_;
    mutable stk::mesh::EntityVector nodeList_;

  private:
    stk::mesh::Ghosting* ghosting_{nullptr};
  };

  class ElemMesh
  {
  public:
    using EntityKey = stk::mesh::EntityKey;
    using EntityProc = stk::search::IdentProc<stk::mesh::EntityKey>;
    using EntityProcVec = std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>;
    using BoundingBox = std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>;

    ElemMesh(stk::mesh::BulkData& bulk, stk::mesh::Selector sel)
    : bulk_(bulk), sel_(sel)
    {}

    stk::ParallelMachine comm() const { return bulk_.parallel(); }


    void update_values()
    {
//      if (ghosting_ != nullptr) {
//        stk::mesh::communicate_field_data(*ghosting_, fields_);
//        stk::mesh::copy_owned_to_shared(bulk_, fields_);
//      }
    }

    void bounding_boxes(std::vector<BoundingBox>& v) const
    {
      v = generate_bounding_boxes_for_elements(bulk_, sel_);
    }
    stk::mesh::BulkData& bulk_;
    mutable std::unordered_map<stk::mesh::EntityKey, std::array<double,3>> pointMap_;
  private:
    stk::mesh::Selector sel_;
    stk::mesh::Ghosting* ghosting_{nullptr};
  };

  class ElementToNodeInterpolation
  {
  public:
    using MeshA = ElemMesh;
    using EntityKeyA = ElemMesh::EntityKey;
    using EntityProcA = ElemMesh::EntityProc;
    using BoundingBoxA = ElemMesh::BoundingBox;

    using MeshB = NodeMesh;
    using EntityKeyB = NodeMesh::EntityKey;
    using EntityProcB = NodeMesh::EntityProc;
    using BoundingBoxB = NodeMesh::BoundingBox;

    using EntityProcRelation = std::pair<EntityProcB, EntityProcA>;
    using EntityProcRelationVec = std::vector<EntityProcRelation>;
    using EntityKeyMap = std::unordered_multimap<EntityKeyB, EntityKeyA>;

    static void filter_to_nearest(
      EntityKeyMap& rangeToDomainMap,
      ElemMesh& elemMesh,
      NodeMesh& nodeMesh)
    {
      elemMesh.pointMap_.clear();

      std::vector<double> elemCoordsScratch(27);
      auto& coordField = coord_field(elemMesh.bulk_);
      auto currentKeyIterator = rangeToDomainMap.begin();
      while (currentKeyIterator != rangeToDomainMap.end())
      {
        auto rangeKey = currentKeyIterator->first;

        const double* const pointCoords = stk::mesh::field_data(coordField, nodeMesh.bulk_.get_entity(rangeKey));
        auto keys = rangeToDomainMap.equal_range(rangeKey);
        auto nearest = keys.second;

        double bestX = std::numeric_limits<double>::max();
        for (auto ii = keys.first; ii != keys.second; ++ii) {
          const auto theBox = ii->second;
          auto elem = elemMesh.bulk_.get_entity(theBox);
          auto& me = *MasterElementRepo::get_surface_master_element(elemMesh.bulk_.bucket(elem).topology());

          std::array<double,3> parametricCoords;
          double parametricDistance;
          const auto* elem_nodes = nodeMesh.bulk_.begin_nodes(elemMesh.bulk_.get_entity(theBox));
          std::tie(parametricCoords, parametricDistance) = parametric_coords_for_point(
            me,
            elem_nodes,
            elemCoordsScratch,
            coord_field(nodeMesh.bulk_),
            pointCoords
          );

          if (parametricDistance < bestX) {
            bestX = parametricDistance;
            elemMesh.pointMap_[currentKeyIterator->first] = parametricCoords;
            nearest = ii;
          }
        }
        currentKeyIterator = keys.second;
        if (nearest != keys.first) rangeToDomainMap.erase(keys.first, nearest);
        if (nearest != keys.second) rangeToDomainMap.erase(++nearest, keys.second);
      }
    }

    static void apply(
      NodeMesh& nodeMesh,
      ElemMesh& elemMesh,
      EntityKeyMap& rangeToDomainMap)
    {
      std::vector<double> elemFieldScratch(8);
      for (auto& pointElemKeyPair : rangeToDomainMap) {
        auto node = nodeMesh.bulk_.get_entity(pointElemKeyPair.first);
        auto elem = elemMesh.bulk_.get_entity(pointElemKeyPair.second);
        auto& me = *MasterElementRepo::get_surface_master_element(elemMesh.bulk_.bucket(elem).topology());

        const auto& fields = nodeMesh.bulk_.mesh_meta_data().get_fields(stk::topology::NODE_RANK);
        const auto* elem_nodes = elemMesh.bulk_.begin_nodes(elem);

        for (auto field : fields) {
          if (elemMesh.bulk_.mesh_meta_data().get_field(stk::topology::NODE_RANK, field->name()) != nullptr
           && field->name() != "coordinates")
          {
            double* field_data = static_cast<double*>(stk::mesh::field_data(*field, node));
            interpolate_point(
              me,
              elem_nodes,
              elemFieldScratch,
              *field,
              elemMesh.pointMap_.at(pointElemKeyPair.first).data(),
              field_data
            );
          }
        }
      }
    }
  };

  stk::mesh::EntityVector generate_nodes_from_point_list(
    stk::mesh::BulkData& bulk,
    std::vector<stk::search::Point<double>> xlocs)
  {
    stk::mesh::EntityVector nodeList(xlocs.size());
    bulk.modification_begin();
    stk::mesh::EntityIdVector ids;
    bulk.generate_new_ids(stk::topology::NODE_RANK, xlocs.size(), ids);
    nodeList.resize(xlocs.size());
    for (unsigned k = 0; k < xlocs.size(); ++k) {
      nodeList[k] = bulk.declare_node(ids[k]);
    }
    bulk.modification_end();

    auto& coordField = coord_field(bulk);
    const int dim = bulk.mesh_meta_data().spatial_dimension();
    for (unsigned k = 0; k < xlocs.size(); ++k) {
      double* coord = stk::mesh::field_data(coordField, nodeList[k]);
      for (int d = 0; d < dim; ++d) {
        coord[d] = xlocs[k][d];
      }
    }
    return nodeList;
  }

  stk::transfer::GeometricTransfer<ElementToNodeInterpolation> create_element_to_node_interpolative_transfer(
    stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& elemSelector,
    const stk::mesh::EntityVector& nodeList)
  {
    auto elemMesh = boost::make_shared<ElemMesh>(bulk, elemSelector);
    auto nodeMesh = boost::make_shared<NodeMesh>(bulk, nodeList);

    return stk::transfer::GeometricTransfer<ElementToNodeInterpolation>(elemMesh, nodeMesh, "elem_node_transfer" );
  }

  void append_nodes_to_transfer(
    stk::transfer::GeometricTransfer<ElementToNodeInterpolation>& transfer,
    const stk::mesh::EntityVector& nodeList)
  {
    transfer.meshb()->nodeList_.insert(transfer.meshb()->nodeList_.end(), nodeList.begin(), nodeList.end());
  }

  std::vector<stk::mesh::EntityProc> stk_mesh_entity_proc_vector(
    stk::mesh::BulkData& bulk,
    std::vector<stk::search::IdentProc<stk::mesh::EntityKey>> searchKeys)
  {
    std::vector<stk::mesh::EntityProc> entKeys(searchKeys.size());
    for (auto& searchKey : searchKeys) {
      entKeys.emplace_back(bulk.get_entity(searchKey.id()), searchKey.proc());
    }
    stk::util::sort_and_unique(entKeys);
    return entKeys;
  }

  void update_ghosting(
    stk::mesh::BulkData& bulk,
    stk::mesh::Ghosting* transferGhosting,
    const std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>& entity_keys)
  {
    if(!is_empty_on_all_procs(bulk.parallel(), entity_keys)) {
      bulk.modification_begin();
      {
        if (nullptr == transferGhosting) {
          transferGhosting = &bulk.create_ghosting("transfer_ghosting");
        }
        bulk.change_ghosting(*transferGhosting, stk_mesh_entity_proc_vector(bulk, entity_keys));
      }
      bulk.modification_end();
      stk::mesh::communicate_field_data(*transferGhosting, { bulk.mesh_meta_data().coordinate_field() });
    }
  }
  template <typename TransferType>
  void setup_transfer(stk::transfer::GeometricTransfer<ElementToNodeInterpolation>& transfer)
  {
    transfer.coarse_search();
    std::vector<stk::search::IdentProc<stk::mesh::EntityKey>> entity_keys;
    transfer.determine_entities_to_copy(entity_keys);
    stk::mesh::Ghosting* ghosting = nullptr;
    update_ghosting(transfer.mesha()->bulk_, ghosting, entity_keys);
    transfer.local_search();
  }

  stk::transfer::GeometricTransfer<ElementToNodeInterpolation> generate_transfer(
    stk::mesh::BulkData& bulk,
    stk::mesh::Selector& elemSelector,
    stk::mesh::EntityVector nodeList)
  {
    auto transfer = create_element_to_node_interpolative_transfer(bulk, elemSelector, nodeList);
    transfer.coarse_search();
    std::vector<stk::search::IdentProc<stk::mesh::EntityKey>> entity_keys;
    transfer.determine_entities_to_copy(entity_keys);
    stk::mesh::Ghosting* ghosting = nullptr;
    update_ghosting(bulk, ghosting, entity_keys);
    transfer.local_search();

    return transfer;
  }

  void transfer(stk::transfer::GeometricTransfer<ElementToNodeInterpolation>& transfer, stk::mesh::BulkData& bulk)
  {
    if (transfer.mesha()->pointMap_.empty()) {
      transfer.coarse_search();
      std::vector<stk::search::IdentProc<stk::mesh::EntityKey>> entity_keys;
      transfer.determine_entities_to_copy(entity_keys);
      stk::mesh::Ghosting* ghosting = nullptr;
      update_ghosting(bulk, ghosting, entity_keys);
      transfer.local_search();
    }
    transfer.apply();
  }

//==========================================================================
// Class Definition
//==========================================================================
// Transfer - base class for Transfer
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
Transfer::Transfer(
  Transfers &transfers)
  : transfers_(transfers),
    couplingPhysicsSpecified_(false),
    transferVariablesSpecified_(false),
    couplingPhysicsName_("none"),
    fromRealm_(NULL),
    toRealm_(NULL),
    name_("none"),
    transferType_("none"),
    transferObjective_("multi_physics"),
    searchMethodName_("none"),
    searchTolerance_(1.0e-4),
    searchExpansionFactor_(1.5)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
Transfer::~Transfer()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- load ------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::load(const YAML::Node & node)
{

  name_ = node["name"].as<std::string>() ;
  transferType_ = node["type"].as<std::string>() ;
  if ( node["objective"] ) {
    transferObjective_ = node["objective"].as<std::string>() ;
  }

  if ( node["coupling_physics"] ) {
    couplingPhysicsName_ = node["coupling_physics"].as<std::string>() ;
    couplingPhysicsSpecified_ = true;
  }

  // realm names
  const YAML::Node realmPair = node["realm_pair"];
  if ( realmPair.size() != 2 )
    throw std::runtime_error("XFER::Error: need two realm pairs for xfer");
  realmPairName_.first = realmPair[0].as<std::string>() ;
  realmPairName_.second = realmPair[1].as<std::string>() ;

  // set bools for variety of mesh part declarations
  const bool hasOld = node["mesh_part_pair"];
  const bool hasNewFrom = node["from_target_name"];
  const bool hasNewTo = node["to_target_name"];

  // mesh part pairs
  if ( hasOld ) {
    // error check to ensure old and new are not mixed
    if ( hasNewFrom || hasNewTo )
      throw std::runtime_error("XFER::Error: part definition error: can not mix mesh part line commands");

    // proceed safely
    const YAML::Node meshPartPairName = node["mesh_part_pair"];
    if ( meshPartPairName.size() != 2 )
      throw std::runtime_error("need two mesh part pairs for xfer");
    // resize and set the value
    fromPartNameVec_.resize(1);
    toPartNameVec_.resize(1);
    fromPartNameVec_[0] = meshPartPairName[0].as<std::string>() ;
    toPartNameVec_[0] = meshPartPairName[1].as<std::string>() ;
  }
  else {
    // new methodology that allows for full target; error check
    if ( !hasNewFrom )
      throw  std::runtime_error("XFER::Error: part definition error: missing a from_target_name");
    if ( !hasNewTo )
      throw  std::runtime_error("XFER::Error: part definition error: missing a to_target_name");

    // proceed safely; manage "from" parts
    const YAML::Node targetsFrom = node["from_target_name"];
    if (targetsFrom.Type() == YAML::NodeType::Scalar) {
      fromPartNameVec_.resize(1);
      fromPartNameVec_[0] = targetsFrom.as<std::string>();
    }
    else {
      fromPartNameVec_.resize(targetsFrom.size());
      for (size_t i=0; i < targetsFrom.size(); ++i) {
        fromPartNameVec_[i] = targetsFrom[i].as<std::string>();
      }
    }
    
    // manage "to" parts
    const YAML::Node &targetsTo = node["to_target_name"];
    if (targetsTo.Type() == YAML::NodeType::Scalar) {
      toPartNameVec_.resize(1);
      toPartNameVec_[0] = targetsTo.as<std::string>();
    }
    else {
      toPartNameVec_.resize(targetsTo.size());
      for (size_t i=0; i < targetsTo.size(); ++i) {
        toPartNameVec_[i] = targetsTo[i].as<std::string>();
      }
    }
  }

  // search method
  if ( node["search_method"] ) {
    searchMethodName_ = node["search_method"].as<std::string>() ;
  }

  // search tolerance which forms the initail size of the radius
  if ( node["search_tolerance"] ) {
    searchTolerance_ = node["search_tolerance"].as<double>() ;
  }

  // search expansion factor if points are not found
  if ( node["search_expansion_factor"] ) {
    searchExpansionFactor_ = node["search_expansion_factor"].as<double>() ;
  }

  // now possible field names
  const YAML::Node y_vars = node["transfer_variables"];
  if (y_vars) {
    transferVariablesSpecified_ = true;
    std::string fromName, toName;
    for (size_t ioption = 0; ioption < y_vars.size(); ioption++) {
      const YAML::Node y_var = y_vars[ioption] ;
      size_t varPairSize = y_var.size();
      if ( varPairSize != 2 )
        throw std::runtime_error("need two field name pairs for xfer");
      fromName = y_var[0].as<std::string>() ;
      toName = y_var[1].as<std::string>() ;
      transferVariablesPairName_.push_back(std::make_pair(fromName, toName));
    }
    
    // warn the user...
    NaluEnv::self().naluOutputP0()
      << "Specifying the transfer variables requires expert understanding; consider using coupling_physics" << std::endl;
  }

  // sanity check
  if ( couplingPhysicsSpecified_ && transferVariablesSpecified_ )
    throw std::runtime_error("physics set and transfer variables specified; will go with variables specified");

  if ( !couplingPhysicsSpecified_ && !transferVariablesSpecified_ )
    throw std::runtime_error("neither physics set nor transfer variables specified");

  // now proceed with possible pre-defined transfers
  if ( !transferVariablesSpecified_ ) {
    // hard code for a single type of physics transfer
    if ( couplingPhysicsName_ == "fluids_cht" ) {
      // h
      std::pair<std::string, std::string> thePairH;
      std::string sameNameH = "heat_transfer_coefficient";
      thePairH = std::make_pair(sameNameH, sameNameH);
      transferVariablesPairName_.push_back(thePairH);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string sameNameT = "reference_temperature";
      thePairT = std::make_pair(sameNameT, sameNameT);
      transferVariablesPairName_.push_back(thePairT);
    }
    else if ( couplingPhysicsName_ == "fluids_robin" ) {
      // q
      std::pair<std::string, std::string> thePairQ;
      std::string sameNameQ = "normal_heat_flux";
      thePairQ = std::make_pair(sameNameQ, sameNameQ);
      transferVariablesPairName_.push_back(thePairQ);
      // Too
      std::pair<std::string, std::string> thePairT;
      std::string fluidsT  = "temperature";
      std::string thermalT = "reference_temperature";
      thePairT = std::make_pair(fluidsT, thermalT);
      transferVariablesPairName_.push_back(thePairT);
      // alpha
      std::pair<std::string, std::string> thePairA;
      std::string sameNameA = "robin_coupling_parameter";
      thePairA = std::make_pair(sameNameA, sameNameA);
      transferVariablesPairName_.push_back(thePairA);
    }
    else if ( couplingPhysicsName_ == "thermal_cht" ) {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    }
    else if ( couplingPhysicsName_ == "thermal_robin" ) {
      // T -> T
      std::pair<std::string, std::string> thePairT;
      std::string temperatureName = "temperature";
      thePairT = std::make_pair(temperatureName, temperatureName);
      transferVariablesPairName_.push_back(thePairT);
      // T -> Tbc
      std::pair<std::string, std::string> thePairTbc;
      std::string temperatureBcName = "temperature_bc";
      thePairTbc = std::make_pair(temperatureName, temperatureBcName);
      transferVariablesPairName_.push_back(thePairTbc);
    }
    else {
      throw std::runtime_error("only supports pre-defined fluids/thermal_cht/robin; perhaps you can use the generic interface");
    }
  }

}

//--------------------------------------------------------------------------
//-------- breadboard ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::breadboard()
{
  // realm pair
  const std::string fromRealmName = realmPairName_.first;
  const std::string toRealmName = realmPairName_.second;

  // extact the realms
  fromRealm_ = root()->realms_->find_realm(fromRealmName);
  if ( NULL == fromRealm_ )
    throw std::runtime_error("from realm in xfer is NULL");
  toRealm_ = root()->realms_->find_realm(toRealmName);
  if ( NULL == toRealm_ )
    throw std::runtime_error("to realm in xfer is NULL");

  // advertise this transfer to realm; for calling control
  fromRealm_->augment_transfer_vector(this, transferObjective_, toRealm_);
 
  // meta data; bulk data to early to extract?
  stk::mesh::MetaData &fromMetaData = fromRealm_->meta_data();
  stk::mesh::MetaData &toMetaData = toRealm_->meta_data();

  // from mesh parts..
  for ( size_t k = 0; k < fromPartNameVec_.size(); ++k ) {
    // get the part; no need to subset
    stk::mesh::Part *fromTargetPart = fromMetaData.get_part(fromPartNameVec_[k]);
    if ( NULL == fromTargetPart )
      throw std::runtime_error("from target part in xfer is NULL; check: " + fromPartNameVec_[k]);
    else
      fromPartVec_.push_back(fromTargetPart);
  }

  // to mesh parts
  for ( size_t k = 0; k < toPartNameVec_.size(); ++k ) {
    // get the part; no need to subset
    stk::mesh::Part *toTargetPart = toMetaData.get_part(toPartNameVec_[k]);
    if ( NULL == toTargetPart )
      throw std::runtime_error("to target part in xfer is NULL; check: " + toPartNameVec_[k]);
    else
      toPartVec_.push_back(toTargetPart);
  }

  // could extract the fields from the realm now and save them off?... 
  // FIXME: deal with STATE....

  // output
  const bool doOutput = true;
  if ( doOutput ) {

    // realm names
    NaluEnv::self().naluOutputP0() << "Xfer Setup Information: " << name_ << std::endl;
    NaluEnv::self().naluOutputP0() << "the From realm name is: " << fromRealm_->name_ << std::endl;
    NaluEnv::self().naluOutputP0() << "the To realm name is: " << toRealm_->name_ << std::endl;

    // provide mesh part names for the user
    NaluEnv::self().naluOutputP0() << "From/To Part Review: " << std::endl;
    for ( size_t k = 0; k < fromPartVec_.size(); ++k )
      NaluEnv::self().naluOutputP0() << "the From mesh part name is: " << fromPartVec_[k]->name() << std::endl;
    for ( size_t k = 0; k < toPartVec_.size(); ++k )
      NaluEnv::self().naluOutputP0() << "the To mesh part name is: " << toPartVec_[k]->name() << std::endl;
    
    // provide field names
    for( std::vector<std::pair<std::string, std::string> >::const_iterator i_var = transferVariablesPairName_.begin();
	 i_var != transferVariablesPairName_.end(); ++i_var ) {
      const std::pair<std::string, std::string> thePair = *i_var;
      NaluEnv::self().naluOutputP0() << "From variable " << thePair.first << " To variable " << thePair.second << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
//-------- allocate_stk_transfer -------------------------------------------
//--------------------------------------------------------------------------
void Transfer::allocate_stk_transfer() {

  const stk::mesh::MetaData    &fromMetaData = fromRealm_->meta_data();
        stk::mesh::BulkData    &fromBulkData = fromRealm_->bulk_data();
  const std::string            &fromcoordName   = fromRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string> > &FromVar = transferVariablesPairName_;
  const stk::ParallelMachine    &fromComm    = fromRealm_->bulk_data().parallel();

  boost::shared_ptr<FromMesh >
    from_mesh (new FromMesh(fromMetaData, fromBulkData, *fromRealm_, fromcoordName, FromVar, fromPartVec_, fromComm));

  stk::mesh::MetaData    &toMetaData = toRealm_->meta_data();
  stk::mesh::BulkData    &toBulkData = toRealm_->bulk_data();
  const std::string             &tocoordName   = toRealm_->get_coordinates_name();
  const std::vector<std::pair<std::string, std::string> > &toVar = transferVariablesPairName_;
  const stk::ParallelMachine    &toComm    = toRealm_->bulk_data().parallel();

  boost::shared_ptr<ToMesh >
    to_mesh (new ToMesh(toMetaData, toBulkData, *toRealm_, tocoordName, toVar, toPartVec_, toComm, searchTolerance_));

  typedef stk::transfer::GeometricTransfer< class LinInterp< class FromMesh, class ToMesh > > STKTransfer;

  // extract search type
  stk::search::SearchMethod searchMethod = stk::search::KDTREE;
  if ( searchMethodName_ == "boost_rtree" )
    searchMethod = stk::search::BOOST_RTREE;
  else if ( searchMethodName_ == "stk_kdtree" )
    searchMethod = stk::search::KDTREE;
  else
    NaluEnv::self().naluOutputP0() << "Transfer::search method not declared; will use stk_kdtree" << std::endl;
  transfer_.reset(new STKTransfer(from_mesh, to_mesh, name_, searchExpansionFactor_, searchMethod));
}

//--------------------------------------------------------------------------
//-------- ghost_from_elements ---------------------------------------------
//--------------------------------------------------------------------------
void Transfer::ghost_from_elements()
{
  typedef stk::transfer::GeometricTransfer< class LinInterp< class FromMesh, class ToMesh > > STKTransfer;

  const boost::shared_ptr<STKTransfer> transfer =
      boost::dynamic_pointer_cast<STKTransfer>(transfer_);
  const boost::shared_ptr<STKTransfer::MeshA> mesha = transfer->mesha();

  STKTransfer::MeshA::EntityProcVec entity_keys;
  transfer->determine_entities_to_copy(entity_keys);
  mesha->update_ghosting(entity_keys);
}

//--------------------------------------------------------------------------
//-------- initialize_begin ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_begin()
{
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::initialize_begin() for: " << name_ << std::endl;
  double time = -NaluEnv::self().nalu_time();
  allocate_stk_transfer();
  transfer_->coarse_search();
  time += NaluEnv::self().nalu_time();
  fromRealm_->timerTransferSearch_ += time;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::change_ghosting()
{
  ghost_from_elements();
}

//--------------------------------------------------------------------------
//-------- initialize_end ------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::initialize_end()
{
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::initialize_end() for: " << name_ << std::endl;
  transfer_->local_search();
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
Transfer::execute()
{
  // do the xfer
  NaluEnv::self().naluOutputP0() << std::endl;
  NaluEnv::self().naluOutputP0() << "PROCESSING Transfer::execute() for: " << name_ << std::endl;

  // provide field names
  for( std::vector<std::pair<std::string, std::string> >::const_iterator i_var = transferVariablesPairName_.begin();
       i_var != transferVariablesPairName_.end(); ++i_var ) {
    const std::pair<std::string, std::string> thePair = *i_var;
    NaluEnv::self().naluOutputP0() << "XFER From variable: " << thePair.first << " To variable " << thePair.second << std::endl;
  }
  NaluEnv::self().naluOutputP0() << std::endl;
  transfer_->apply();
}

Simulation *Transfer::root() { return parent()->root(); }
Transfers *Transfer::parent() { return &transfers_; }

} // namespace nalu
} // namespace Sierra
