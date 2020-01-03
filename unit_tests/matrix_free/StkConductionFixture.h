#ifndef STK_CONDUCTION_FIXTURE_H
#define STK_CONDUCTION_FIXTURE_H

#include "Tpetra_Map.hpp"

#include "gtest/gtest.h"
#include "mpi.h"

#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_ngp/Ngp.hpp"
#include "stk_ngp/NgpFieldManager.hpp"
#include "stk_topology/topology.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/CoordinateMapping.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/Hex27Fixture.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/HexFixture.hpp"

class ConductionFixture : public ::testing::Test
{
protected:
  static constexpr int order = 1;
  ConductionFixture(int nx, double scale);
  stk::mesh::Field<double, stk::mesh::Cartesian3d>& coordinate_field();
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  stk::io::StkMeshIoBroker io;
  stk::mesh::Field<double>& q_field;
  stk::mesh::Field<double>& qbc_field;
  stk::mesh::Field<double>& flux_field;
  stk::mesh::Field<double>& qtmp_field;
  stk::mesh::Field<double>& alpha_field;
  stk::mesh::Field<double>& lambda_field;
  stk::mesh::Field<stk::mesh::EntityId>& gid_field;
  stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
    tpetra_gid_field;

  ngp::Mesh mesh;
  ngp::FieldManager fm;
  ngp::Field<stk::mesh::EntityId> gid_field_ngp;
};

class ConductionFixtureP2 : public ::testing::Test
{
protected:
  static constexpr int order = 2;
  ConductionFixtureP2(int nx, double scale);
  stk::mesh::Field<double, stk::mesh::Cartesian3d>& coordinate_field();
  stk::mesh::fixtures::Hex27Fixture fixture;
  stk::mesh::MetaData& meta;
  stk::mesh::BulkData& bulk;
  stk::io::StkMeshIoBroker io;
  stk::mesh::Field<double>& q_field;
  stk::mesh::Field<double>& qbc_field;
  stk::mesh::Field<double>& flux_field;
  stk::mesh::Field<double>& qtmp_field;
  stk::mesh::Field<double>& alpha_field;
  stk::mesh::Field<double>& lambda_field;
  stk::mesh::Field<stk::mesh::EntityId>& gid_field;
  stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
    tpetra_gid_field;

  ngp::Mesh mesh;
  ngp::FieldManager fm;
  ngp::Field<stk::mesh::EntityId> gid_field_ngp;
};
#endif
