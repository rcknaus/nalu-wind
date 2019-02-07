/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <xfer/ElemToNodeTransfer.h>

#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <NaluEnv.h>
#include <nalu_make_unique.h>

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
#include <stk_io/StkMeshIoBroker.hpp>


#include <stk_mesh/base/GetEntities.hpp>

#include <stk_mesh/base/HashEntityAndEntityKey.hpp>
// stk_search
#include <stk_search/SearchMethod.hpp>

#include "boost/make_shared.hpp"

namespace sierra{
namespace nalu{
namespace transfer {

namespace {

constexpr int max_dim = 3;

const VectorFieldType& coord_field(const stk::mesh::BulkData& bulk)
{
  return *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
}

std::pair<std::array<double,3>, double>
parametric_coords_for_point(
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
    const double* const fieldData = static_cast<const double* const>(stk::mesh::field_data(field, elem_nodes[n]));
    for (int d = 0; d < fieldSize; ++d) {
      elemFieldScratch[nNodes * d + n] = fieldData[d];
    }
  }
  me.interpolatePoint(fieldSize, parametricCoords, elemFieldScratch.data(), values);
}

std::vector<std::pair<stk::search::Sphere<double>, stk::search::IdentProc<stk::mesh::EntityKey>>>
generate_bounding_spheres_for_nodes(
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

bool is_empty_on_all_procs(stk::ParallelMachine comm, const std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>& vec) {
  const int isEmptyLocal = vec.empty() ? 1 : 0;
  int isEmptyGlobal = 0;
  stk::all_reduce_sum(comm, &isEmptyLocal, &isEmptyGlobal, 1);
  return (isEmptyGlobal == 0);
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

struct SearchOptions
{
  double initialRadius{1.0e-3};
  double expansionFactor{1.5};
  stk::search::SearchMethod method{stk::search::KDTREE};
};


class NodeMesh
{
public:
 using EntityKey = stk::mesh::EntityKey;
 using EntityProc = stk::search::IdentProc<stk::mesh::EntityKey>;
 using EntityProcVec = std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>;
 using BoundingBox = std::pair<stk::search::Sphere<double>, stk::search::IdentProc<stk::mesh::EntityKey>>;

 NodeMesh(stk::mesh::BulkData& bulk, const stk::mesh::Selector& sel, stk::mesh::FieldVector fields = {});
 NodeMesh(stk::mesh::BulkData& bulk, stk::mesh::EntityVector nodeList, stk::mesh::FieldVector fields = {});

 stk::ParallelMachine comm() const { return bulk_.parallel(); }

 void update_values();

 void bounding_boxes(std::vector<BoundingBox>& v) const;


 stk::mesh::BulkData& bulk() { return bulk_;}
 const stk::mesh::BulkData& bulk() const { return bulk_; }

 const stk::mesh::EntityVector& node_list() const { return nodeList_; }
 stk::mesh::EntityVector& node_list() { return nodeList_; }

 const stk::mesh::FieldVector& fields() const { return fields_;}
 stk::mesh::FieldVector& fields() { return fields_;}

 void set_fields(stk::mesh::FieldVector fields);

private:
 stk::mesh::BulkData& bulk_;
 stk::mesh::EntityVector nodeList_;
 stk::mesh::FieldVector fields_;

 SearchOptions searchOptions_;
};

class ElemMesh
{
public:
 using EntityKey = stk::mesh::EntityKey;
 using EntityProc = stk::search::IdentProc<stk::mesh::EntityKey>;
 using EntityProcVec = std::vector<stk::search::IdentProc<stk::mesh::EntityKey>>;
 using BoundingBox = std::pair<stk::search::Box<double>, stk::search::IdentProc<stk::mesh::EntityKey>>;

 ElemMesh(stk::mesh::BulkData& bulk, const stk::mesh::Selector& sel, stk::mesh::FieldVector fields = {});


 ~ElemMesh()
 {
   if (ghosting_ != nullptr) {
     bulk_.modification_begin();
     bulk_.destroy_ghosting(*ghosting_);
     bulk_.modification_end();
   }
 }

 stk::ParallelMachine comm() const { return bulk_.parallel(); }

 void update_values();

 void bounding_boxes(std::vector<BoundingBox>& v) const;


 stk::mesh::BulkData& bulk() { return bulk_; }
 const stk::mesh::BulkData& bulk() const { return bulk_; }

 const stk::mesh::Selector& elem_selector() const { return sel_; }

 void set_fields(stk::mesh::FieldVector fields);

 std::unordered_map<stk::mesh::EntityKey, std::array<double, max_dim>>& point_map() { return pointMap_; }

private:
 stk::mesh::BulkData& bulk_;
 const stk::mesh::Selector& sel_;
 stk::mesh::Ghosting* ghosting_{nullptr};

 mutable std::unordered_map<stk::mesh::EntityKey, std::array<double, max_dim>> pointMap_;
 mutable stk::mesh::FieldVector fields_;
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

 static void filter_to_nearest(EntityKeyMap& rangeToDomainMap, ElemMesh& elemMesh, NodeMesh& nodeMesh);
 static void apply(NodeMesh& nodeMesh, ElemMesh& elemMesh, EntityKeyMap& rangeToDomainMap);
};


ElemMesh::ElemMesh(stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, stk::mesh::FieldVector fields)
: bulk_(bulk), sel_(selector), fields_(fields.empty() ? bulk.mesh_meta_data().get_fields(stk::topology::NODE_RANK) : fields)
{
}

NodeMesh::NodeMesh(stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, stk::mesh::FieldVector fields)
: bulk_(bulk), fields_(fields.empty() ? bulk.mesh_meta_data().get_fields(stk::topology::NODE_RANK) : fields)
{
  stk::mesh::get_selected_entities(selector, bulk.get_buckets(stk::topology::NODE_RANK, selector), nodeList_);
  ThrowRequire(!nodeList_.empty());
}

NodeMesh::NodeMesh(stk::mesh::BulkData& bulk, stk::mesh::EntityVector nodeList, stk::mesh::FieldVector fields)
: bulk_(bulk),
  nodeList_(std::move(nodeList)),
  fields_(fields.empty() ? bulk.mesh_meta_data().get_fields(stk::topology::NODE_RANK) : fields)
{

}

void NodeMesh::bounding_boxes(std::vector<BoundingBox>& v) const
{
  v = generate_bounding_spheres_for_nodes(bulk_, nodeList_, searchOptions_.initialRadius);
}
void ElemMesh::bounding_boxes(std::vector<BoundingBox>& v) const
{
  v = generate_bounding_boxes_for_elements(bulk_, sel_);
}

void NodeMesh::update_values()
{
  stk::mesh::copy_owned_to_shared(bulk_, std::vector<const stk::mesh::FieldBase*>{fields_.begin(), fields_.end()});
}

void ElemMesh::update_values()
{
  if (ghosting_ != nullptr) {
    auto fields = std::vector<const stk::mesh::FieldBase*>{fields_.begin(), fields_.end()};
    fields.insert(fields.end(), bulk_.mesh_meta_data().coordinate_field());
    stk::mesh::communicate_field_data(*ghosting_, fields);
    stk::mesh::copy_owned_to_shared(bulk_, fields);
  }
}

void ElemMesh::set_fields(stk::mesh::FieldVector fields)
{
  fields_.clear();
  for (const auto* field : fields) {
    auto elemMeshField = bulk_.mesh_meta_data().get_field(stk::topology::NODE_RANK, field->name());
    ThrowRequire(elemMeshField != nullptr);
    ThrowRequire(elemMeshField->number_of_states() >= field->number_of_states());
    fields_.push_back(elemMeshField);
  }
}

void NodeMesh::set_fields(stk::mesh::FieldVector fields)
{
  fields_.clear();
  for (const auto* field : fields) {
    auto elemMeshField = bulk_.mesh_meta_data().get_field(stk::topology::NODE_RANK, field->name());
    ThrowRequire(elemMeshField != nullptr);
    ThrowRequire(elemMeshField->number_of_states() >= field->number_of_states());
    fields_.push_back(elemMeshField);
  }
}

void ElementToNodeInterpolation::filter_to_nearest(
  EntityKeyMap& rangeToDomainMap,
  ElemMesh& elemMesh,
  NodeMesh& nodeMesh)
{
  elemMesh.point_map().clear();
  elemMesh.point_map().reserve(rangeToDomainMap.size());

  std::vector<double> elemCoordsScratch(27);
  auto& elemCoordField = coord_field(elemMesh.bulk());
  auto& nodeCoordField = coord_field(nodeMesh.bulk());
  auto currentKeyIterator = rangeToDomainMap.begin();
  while (currentKeyIterator != rangeToDomainMap.end()) {
    auto rangeKey = currentKeyIterator->first;

    const double* const pointCoords = stk::mesh::field_data(nodeCoordField, nodeMesh.bulk().get_entity(rangeKey));
    auto keys = rangeToDomainMap.equal_range(rangeKey);
    auto nearest = keys.second;

    double bestX = std::numeric_limits<double>::max();
    for (auto it = keys.first; it != keys.second; ++it) {
      const auto theBox = it->second;
      auto elem = elemMesh.bulk().get_entity(theBox);
      auto& me = *MasterElementRepo::get_surface_master_element(elemMesh.bulk().bucket(elem).topology());

      std::array<double, max_dim> parametricCoords;
      double parametricDistance;
      const auto* elem_nodes = elemMesh.bulk().begin_nodes(elemMesh.bulk().get_entity(theBox));
      std::tie(parametricCoords, parametricDistance) = parametric_coords_for_point(
        me,
        elem_nodes,
        elemCoordsScratch,
        elemCoordField,
        pointCoords
      );

      if (parametricDistance < bestX) {
        bestX = parametricDistance;
        elemMesh.point_map()[currentKeyIterator->first] = parametricCoords;
        nearest = it;
      }
    }
    currentKeyIterator = keys.second;
    if (nearest != keys.first) rangeToDomainMap.erase(keys.first, nearest);
    if (nearest != keys.second) rangeToDomainMap.erase(++nearest, keys.second);
  }
  ThrowRequire(!elemMesh.point_map().empty());
}

void ElementToNodeInterpolation::apply(
  NodeMesh& nodeMesh,
  ElemMesh& elemMesh,
  EntityKeyMap& rangeToDomainMap)
{
  std::vector<double> elemFieldScratch(8);
  ThrowRequire(!rangeToDomainMap.empty());
  for (auto& pointElemKeyPair : rangeToDomainMap) {
    const auto node = nodeMesh.bulk().get_entity(pointElemKeyPair.first);
    ThrowRequire(nodeMesh.bulk().is_valid(node));
    const auto elem = elemMesh.bulk().get_entity(pointElemKeyPair.second);
    auto& me = *MasterElementRepo::get_surface_master_element(elemMesh.bulk().bucket(elem).topology());

    const auto* elem_nodes = elemMesh.bulk().begin_nodes(elem);
    ThrowRequire(!nodeMesh.fields().empty());
    for (const auto* field : nodeMesh.fields()) {
      const auto* srcField = elemMesh.bulk().mesh_meta_data().get_field(stk::topology::NODE_RANK, field->name());

      if (field->name() != coord_field(nodeMesh.bulk()).name()) {
          //ThrowRequire(srcField->is_state_valid(state));
          auto& srcFieldj = *srcField;//->field_state(state);
          auto& destFieldj = *field;//->field_state(state);

          double* field_data = static_cast<double*>(stk::mesh::field_data(destFieldj, node));
          ThrowRequireMsg(field_data, "no field data for " + destFieldj.name() + " at " + std::to_string(nodeMesh.bulk().identifier(node)));
          interpolate_point(
            me, elem_nodes, elemFieldScratch, srcFieldj,
            elemMesh.point_map().at(pointElemKeyPair.first).data(),
            field_data
          );
        }
    }
  }
}

stk::transfer::GeometricTransfer<ElementToNodeInterpolation> create_element_to_node_interpolative_transfer(
  stk::mesh::BulkData& bulkSrc,
  const stk::mesh::Selector& elemSelector,
  stk::mesh::BulkData& bulkDest,
  const stk::mesh::Selector& nodeSelector)
{
  auto elemMesh = boost::make_shared<ElemMesh>(bulkSrc, elemSelector);
  auto nodeMesh = boost::make_shared<NodeMesh>(bulkDest, nodeSelector);

  return stk::transfer::GeometricTransfer<ElementToNodeInterpolation>(elemMesh, nodeMesh, "elem_node_transfer");
}

stk::mesh::FieldVector negotiate_fields_based_on_names(const stk::mesh::BulkData& mesha, const stk::mesh::BulkData& meshb)
{
  // selects fields to transfer -- all

  stk::mesh::FieldVector fields;
  auto fieldsA = mesha.mesh_meta_data().get_fields(stk::topology::NODE_RANK);
  auto fieldsB = meshb.mesh_meta_data().get_fields(stk::topology::NODE_RANK);

  struct FieldCompare
  {
    bool operator()(const stk::mesh::FieldBase* a, const stk::mesh::FieldBase* b)
    {
      return a->name() < b->name();
    }
  };
  std::sort(fieldsA.begin(), fieldsA.end(), FieldCompare{});
  std::sort(fieldsB.begin(), fieldsB.end(), FieldCompare{});

  std::set_intersection(fieldsA.begin(), fieldsA.end(),
    fieldsB.begin(), fieldsB.end(),
    std::back_inserter(fields), FieldCompare{}
  );

  return fields;
}

void setup_transfer(stk::transfer::GeometricTransfer<ElementToNodeInterpolation>& transfer)
{
  auto fields = negotiate_fields_based_on_names(transfer.mesha()->bulk(), transfer.meshb()->bulk());
  transfer.mesha()->set_fields(fields);
  transfer.meshb()->set_fields(fields);

  std::string allFields = "Fields to transfer: ";
  for (auto* field : fields) {
   allFields += field->name() + ", ";
  }
  allFields = allFields.substr(0, allFields.size()-2);

  std::cout << allFields << std::endl;

  transfer.coarse_search();
  std::vector<stk::search::IdentProc<stk::mesh::EntityKey>> entity_keys;
  transfer.determine_entities_to_copy(entity_keys);
  stk::mesh::Ghosting* ghosting = nullptr;
  update_ghosting(transfer.mesha()->bulk(), ghosting, entity_keys);
  transfer.local_search();
}

stk::transfer::GeometricTransfer<ElementToNodeInterpolation> generate_transfer(
  stk::mesh::BulkData& bulkSrc,
  const stk::mesh::Selector& elemSelector,
  stk::mesh::BulkData& bulkDest,
  const stk::mesh::Selector& nodeSelector)
{
  auto transfer = create_element_to_node_interpolative_transfer(bulkSrc, elemSelector, bulkDest, nodeSelector);
  setup_transfer(transfer);
  return transfer;
}

} // namespace

void transfer_all(
  stk::mesh::BulkData& bulkSrc,
  const stk::mesh::Selector& elemSelector,
  stk::mesh::BulkData& bulkDest,
  const stk::mesh::Selector& nodeSelector)
{
  generate_transfer(bulkSrc, elemSelector, bulkDest, nodeSelector).apply();
}

std::unique_ptr<stk::transfer::TransferBase> create_element_to_node_transfer(
  stk::mesh::BulkData& bulkSrc,
  const stk::mesh::Selector& elemSelector,
  stk::mesh::BulkData& bulkDest,
  const stk::mesh::Selector& nodeSelector)
{
  auto transfer = generate_transfer(bulkSrc, elemSelector, bulkDest, nodeSelector);
  return std::unique_ptr<stk::transfer::TransferBase>(new decltype(transfer)(transfer));
}

struct FieldInfo {
  std::string name;
  int length;
};

std::pair<std::unique_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>>
read_mesh(std::string meshName, std::set<FieldInfo> fieldInfoSet, stk::ParallelMachine comm, int dim)
{
  auto metaPtr = make_unique<stk::mesh::MetaData>(dim);
  auto bulkPtr = make_unique<stk::mesh::BulkData>(*metaPtr, comm);

  stk::io::StkMeshIoBroker io(comm);
  io.set_bulk_data(*bulkPtr);
  io.add_mesh_database(meshName, stk::io::READ_RESTART);
  io.create_input_mesh();

  for (auto fieldInfo : fieldInfoSet) {
    auto& scalarField = metaPtr->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, fieldInfo.name);
    stk::mesh::put_field_on_mesh(scalarField, metaPtr->universal_part(), fieldInfo.length, nullptr);
    io.add_input_field({*stk::mesh::get_field_by_name(fieldInfo.name, *metaPtr), fieldInfo.name});
  }
  io.populate_bulk_data();

  std::vector<stk::io::MeshField> missingFields;
  io.read_defined_input_fields(1000.0, &missingFields);
  io.populate_field_data();

  return std::make_pair(std::move(metaPtr), std::move(bulkPtr));
}

void transfer_all(
  const stk::mesh::BulkData& meshNew,
  std::string meshName,
  std::set<FieldInfo> fieldInfoSet,
  stk::ParallelMachine comm)
{
  stk::io::StkMeshIoBroker io(comm);
  io.add_mesh_database(meshName, stk::io::READ_RESTART);
  stk::mesh::MetaData meta(3u);
  stk::mesh::BulkData bulk(meta, comm);
  io.set_bulk_data(bulk);

  io.create_input_mesh();

  for (auto fieldInfo : fieldInfoSet) {
    auto& scalarField = meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, fieldInfo.name);
    stk::mesh::put_field_on_mesh(scalarField, io.get_active_selector(), fieldInfo.length, nullptr);
    io.add_input_field({*stk::mesh::get_field_by_name(fieldInfo.name, meta), fieldInfo.name});
  }
  io.populate_bulk_data();

  std::vector<stk::io::MeshField> missingFields;
  io.read_defined_input_fields(1000.0, &missingFields);
  io.populate_field_data();
}



}
} // namespace nalu
} // namespace Sierra
