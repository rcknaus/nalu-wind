#include <property_evaluator/HDF5TablePropAlgorithm.h>
#include <tabular_props/HDF5Table.h>
#include <tabular_props/H5IO.h>
#include <stk_mesh/base/FieldParallel.hpp>
#include "master_element/MasterElement.h"
#include "master_element/MasterElementHO.h"


#include <Algorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace sierra {
namespace nalu {

//============================================================================
HDF5TablePropAlgorithm::HDF5TablePropAlgorithm(
  Realm & realm,
  stk::mesh::Part * part,
  H5IO *fileIO,
  stk::mesh::FieldBase * prop,
  std::string tablePropName,
  std::vector<std::string> &indVarNameVec,
  std::vector<std::string> &indVarTableNameVec,
  const stk::mesh::MetaData &meta_data)
  : Algorithm(realm, part),
    prop_(dynamic_cast<ScalarFieldType*>(prop)),
    tablePropName_(tablePropName),
    indVarTableNameVec_(indVarTableNameVec),
    indVarSize_(indVarNameVec.size()),
    fileIO_( fileIO )
{ 
  ThrowRequire(prop_);
  // extract the independent fields; check if there is one..
  if ( indVarSize_ == 0 )
    throw std::runtime_error("HDF5TablePropAlgorithm: independent variable size is zero:");

  indVar_.resize(indVarSize_);
  for ( size_t k = 0; k < indVarSize_; ++k) {
    ScalarFieldType *indVar = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, indVarNameVec[k]);
    if ( NULL == indVar ) {
      throw std::runtime_error("HDF5TablePropAlgorithm: independent variable not registered:");
    }
    else {
      indVar_[k] = indVar;
    }
  }
  
  // resize some work vectors
  workIndVar_.resize(indVarSize_);
  workZ_.resize(indVarSize_);

  //read in table
  //read_hdf5( );
  table_ = new HDF5Table( fileIO, tablePropName_, indVarNameVec, indVarTableNameVec ) ;


  auto desc = ElementDescription::create(3, 1);
  auto basis =  LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = TensorProductQuadratureRule("GaussLegendre", 2);
  meSCVPtr = std::unique_ptr<HigherOrderHexSCV>(new HigherOrderHexSCV(*desc, basis, quad));

  // provide some output
  NaluEnv::self().naluOutputP0() << "the Following Table Property name will be extracted: " << tablePropName << std::endl;
  for ( size_t k = 0; k < indVarTableNameVec_.size(); ++k ) {
    NaluEnv::self().naluOutputP0() << "using independent variables: " << indVarTableNameVec_[k] << std::endl;
  }
}
//----------------------------------------------------------------------------
HDF5TablePropAlgorithm::~HDF5TablePropAlgorithm()
{
  delete table_;
}
//----------------------------------------------------------------------------

struct BlendedPropEval
{
  BlendedPropEval(HDF5Table& table, double ignitionStart, double ignitionDt)
  : table_(table), ignitionStartTime_(ignitionStart), ignitionDeltaT_(ignitionDt)
  {
    harmonicAverage = table.name() == "density";
    auto inputNames = table.input_names();

    auto it = std::find(inputNames.begin(), inputNames.end(), "mixture_fraction");
    ThrowRequireMsg(it != inputNames.end(), "No mixture fraction in evaluation");
    zIndex_ = std::distance(inputNames.begin(), it);
  }

  double compute_blending_factor(double time) {
    return std::min(1.0, std::max(0.0, 1 - (time - ignitionStartTime_)/ignitionDeltaT_));
  }

  double blended_prop_eval(double time, std::vector<double>& inputs)
  {
    auto blendFac = compute_blending_factor(time);

    auto zval = inputs[zIndex_];
    auto propReal = table_.query(inputs);

    double propOx = 1;
    double propFuel = 1;
    if (blendFac > std::numeric_limits<double>::min()) {
      inputs[zIndex_] = 0;
      propOx = table_.query(inputs);

      inputs[zIndex_] = 1;
      propFuel = table_.query(inputs);

      inputs[zIndex_] = zval;
    }


    return propReal;
//    if (harmonicAverage) {
//      return blendFac/(zval/propFuel + (1-zval)/propOx) + (1-blendFac)*propReal;
//    }
//    return blendFac*(zval*propFuel + (1-zval)*propOx) + (1-blendFac)*propReal;
  }

  HDF5Table& table_;
  double ignitionStartTime_{1};
  double ignitionDeltaT_{1};

  bool harmonicAverage{false};
  int zIndex_{0};
};



void
HDF5TablePropAlgorithm::execute()
{
  if (!realm_.is_output_step() && table_->name() == "temperature") {
    return;
  }
  NaluEnv::self().naluOutputP0() << "Evaluate property: " << table_->name() << " begin" <<std::endl;

  const auto& bulk = realm_.bulk_data();
  const auto& meta = realm_.meta_data();

  auto time = realm_.get_current_time();

  auto propEval = BlendedPropEval(*table_, 0.1, 2.0);


  // make sure that partVec_ is size one
  ThrowAssert( partVec_.size() == 1 );

  ThrowRequire(meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume"));
  ThrowRequire(meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name()));

  auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  auto& dnvField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  HigherOrderHexSCV& meSCV = *meSCVPtr;

  stk::mesh::field_fill(0.0, *prop_);
  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_)
    & (meta.locally_owned_part() | meta.globally_shared_part());

  for (const auto* ib : bulk.get_buckets(stk::topology::ELEM_RANK, selector)) {
    Kokkos::View<double**> workIndVar("ind_var", indVar_.size(), meSCV.nodes_per_element());

    std::vector<double> ipInterpWeights(meSCV.num_integration_points()*meSCV.nodes_per_element());
    meSCV.shape_fcn(ipInterpWeights.data());

    std::vector<double> integrationWeights(meSCV.num_integration_points());

    std::vector<double> coords(meSCV.nodes_per_element() * meSCV.ndim());

    for (auto elem : *ib) {
      const auto* node_rels = bulk.begin_nodes(elem);
      for (int n = 0; n < meSCV.nodes_per_element(); ++n) {

        const auto* coord_data = stk::mesh::field_data(coordField, node_rels[n]);
        for (int d = 0; d < meSCV.ndim(); ++d) {
          coords[meSCV.ndim() * n + d] = coord_data[d];
        }

        for ( size_t l = 0; l < indVarSize_; ++l) {
          workIndVar(l, n) = *stk::mesh::field_data(*indVar_[l], node_rels[n]);
        }
      }

      double err = 0;
      meSCV.determinant(1, coords.data(), integrationWeights.data(), &err);

      for (int ip = 0; ip < meSCV.num_integration_points(); ++ip) {
        const auto* interpWeights = &ipInterpWeights[ip * meSCV.nodes_per_element()];
        for (unsigned l = 0; l < indVarSize_; ++l) {
          workZ_[l] = 0;
          for (int n = 0; n < meSCV.nodes_per_element(); ++n) {
            workZ_[l] += interpWeights[n] * workIndVar(l, n);
          }
        }

        auto& ipNodeMap = meSCV.ip_node_map();
        auto nn = node_rels[ipNodeMap[ip]];
        *stk::mesh::field_data(*prop_, nn) += propEval.blended_prop_eval(time, workZ_) * integrationWeights[ip];
      }
    }
  }

  stk::mesh::parallel_sum(bulk, std::vector<const stk::mesh::FieldBase*>{prop_});

  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    realm_.periodic_field_update(prop_, fieldSize);
  }

  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, selector)) {
    for (auto node : *ib) {
      *stk::mesh::field_data(*prop_, node) /= *stk::mesh::field_data(dnvField, node);
    }
  }


  NaluEnv::self().naluOutputP0() << "Evaluate property: " << table_->name() << "   end" <<std::endl;


}
//============================================================================

} // end nalu namespace
} // end sierra namespace

