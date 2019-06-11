#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include "ConductionInteriorOperator.h"
#include "MatrixFreeOperator.h"
#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeTypes.h"
#include "master_element/DirectionMacros.h"
#include "SimdInterface.h"
#include "KokkosInterface.h"

#include <stk_mesh/base/FieldBLAS.hpp>  // for field_copy, field_fill, etc
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_unit_tests/stk_mesh_fixtures/HexFixture.hpp>

#include "ConductionInteriorOperator.h"
#include "MatrixFreeOperator.h"
#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeTypes.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include <tpetra_linsys/SolutionPointCategorizer.h>
#include <FieldTypeDef.h>
#include <nalu_make_unique.h>

#include "BelosLinearProblem.hpp"

#include "UnitTestUtils.h"

namespace sierra { namespace nalu {

  static constexpr double dt = 1e4;

class TpetraMatrixFreeFixture : public ::testing::Test
{
protected:
  TpetraMatrixFreeFixture()
    : comm(MPI_COMM_WORLD),
      meta(3u), bulk(meta, comm, stk::mesh::BulkData::NO_AUTO_AURA),
      gidField(meta.declare_field<GlobalIdFieldType>(stk::topology::NODE_RANK, "global_id_field")),
      lidField(meta.declare_field<stk::mesh::Field<int>>(stk::topology::NODE_RANK, "local_id_field")),
      qField(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature", 3)),
      qTmpField(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "tmp")),
      rhocpField(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "density")),
      diffusivityField(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "diffusivity"))
  {
  }

  void set_mesh(int in_nx, double scale = M_PI)
  {
    nx = in_nx;
    meshSpec = "generated:" + std::to_string(nx) + "x" + std::to_string(nx) + "x" + std::to_string(nx);

    double oneD = 1.0;
    int32_t one = 1;
    stk::mesh::EntityId one64 = 1;
    stk::mesh::put_field_on_mesh(lidField, meta.universal_part(), 1, &one);
    stk::mesh::put_field_on_mesh(gidField, meta.universal_part(), 3, &one64);
    stk::mesh::put_field_on_mesh(qField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(qTmpField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(rhocpField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(diffusivityField, meta.universal_part(), 1, &oneD);
    unit_test_utils::fill_hex8_mesh(meshSpec, bulk);

    auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
    for (auto ib: bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        auto* coordptr = stk::mesh::field_data(coordField, node);
        coordptr[0] = scale * (coordptr[0] - 0.5 * nx) * 2 / nx;
        coordptr[1] = scale * (coordptr[1] - 0.5 * nx) * 2 / nx;
        coordptr[2] = scale * (coordptr[2] - 0.5 * nx) * 2 / nx;
      }
    }
    for (auto ib: bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coordField, node);
        *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), node) = std::cos(coordptr[0]);
        *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateN), node) = std::cos(coordptr[0]);
        *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNM1), node) = std::cos(coordptr[0]);
        *stk::mesh::field_data(qTmpField, node) = 0;
        *stk::mesh::field_data(rhocpField, node) = 1.0;
        *stk::mesh::field_data(diffusivityField, node) = 1.0;
        *stk::mesh::field_data(lidField, node) = node.local_offset();
        *stk::mesh::field_data(gidField, node) = bulk.identifier(node);
      }
    }
  }

  stk::ParallelMachine comm;
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  GlobalIdFieldType& gidField;
  stk::mesh::Field<int>& lidField;
  ScalarFieldType& qField;
  ScalarFieldType& qTmpField;
  ScalarFieldType& rhocpField;
  ScalarFieldType& diffusivityField;

  int nx = 1;
  std::string meshSpec;
};

constexpr int default_mesh_size = 64;

using TpetraOperator = Tpetra::Operator<double, local_ordinal_type, global_ordinal_type>;
using TpetraVector = Tpetra::MultiVector<double, local_ordinal_type, global_ordinal_type>;

namespace {

void print_local_data(const TpetraVector& x, const TpetraVector& y)
{
  const auto x_view = x.getLocalView<HostSpace>();
  const auto y_view = y.getLocalView<HostSpace>();
  ThrowRequire(x_view.extent_int(0) == y_view.extent_int(0));
  for (int k = 0; k < y_view.extent_int(0); ++k) {
    std::cout << k << "(x,y): (" << x_view(k,0) << ", " << y_view(k,0)  << ")" << std::endl;
  }
}


} // namespace

TEST_F(TpetraMatrixFreeFixture, check_matrix_free_laplacian_is_zero_for_constant_data)
{
  constexpr int p = 1;
  set_mesh(default_mesh_size);
  const auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())){
    for (const auto node : *ib) {
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), node) = 1;
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateN), node) = 1;
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNM1), node) = 1;
    }
  }

  auto entToLid = serial_entity_offset_to_row_lid_map(bulk, meta.universal_part());
  auto mfInterior = ConductionInteriorOperator<p>(bulk, meta.universal_part(), coordField, rhocpField, diffusivityField, entToLid);
  auto mfBdry = NoOperator();
  auto mfOp = MFOperator<decltype(mfInterior), decltype(mfBdry)>(mfInterior, mfBdry, operator_map(bulk, gidField, meta.universal_part(),{},{}));

  auto x = TpetraVector(mfOp.getDomainMap(),1);
  x.putScalar(0.0);
  auto y = TpetraVector(mfOp.getRangeMap(), 1);
  y.putScalar(0.0);

  mfOp.apply(x, y);

  auto xv = x.getLocalView<HostSpace>();
  auto yv = y.getLocalView<HostSpace>();

  for (int k = 0; k < yv.extent_int(0); ++k) {
    ASSERT_DOUBLE_EQ(yv(k,0), 0);
  }
}

//unit_test_utils::NaluTest& naluObj, const std::string& meshSpec)
//{
//  sierra::nalu::Realm& realm = naluObj.create_realm();
//  realm.setup_nodal_fields();
//  unit_test_utils::fill_hex8_mesh(meshSpec, realm.bulk_data());
//  realm.set_global_id();
//  return realm;
//}

TEST_F(TpetraMatrixFreeFixture, solve_laplacian)
{
  const double scale_fac = 1.0;
  set_mesh(64, scale_fac);
  int len = 0;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part()))
  {
    len += ib->size();
  }

  std::cout << "Mesh Generated ... " << len << std::endl;

  auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part()))
  {
    for (const auto node : *ib)
    {
      const double x = stk::mesh::field_data(coordField, node)[0];
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), node) = std::cos(M_PI*x/scale_fac);
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateN), node) =  std::cos(M_PI*x/scale_fac);
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNM1), node) = std::cos(M_PI*x/scale_fac);

      *stk::mesh::field_data(rhocpField, node) = 1.0e3;
      *stk::mesh::field_data(diffusivityField, node) = 1.0;
    }
  }

  constexpr int p = 1;

  using interior_operator_type = ConductionInteriorOperator<p>;
  using mfop_type = MFOperator<interior_operator_type, NoOperator>;

  std::cout << "Creating operator ..." <<std::endl;
//  auto entToLid = entity_offset_to_row_lid_map(bulk, meta.universal_part(), gidField, {});
  auto entToLid = serial_entity_offset_to_row_lid_map(bulk, meta.universal_part());
  double tstart = MPI_Wtime();
  auto mfInterior = ConductionInteriorOperator<p>(bulk, meta.universal_part(), coordField, rhocpField, diffusivityField, entToLid);
  mfInterior.set_gamma({{0, 0, 0}});
  auto mfBdry = NoOperator();
  double time = MPI_Wtime() - tstart;
  std::cout << "operator creation time: " << time << std::endl;
  auto mfOp = make_rcp<mfop_type>(mfInterior, mfBdry, operator_map(bulk, gidField, meta.universal_part(),{},{}));

//  auto mfProb = MatrixFreeProblem(mfOp, 1);
  TpetraMatrixFreeSolver solver(1);
  MatrixFreeProblem mfProb(mfOp,1);
  auto sln = mfProb.sln;
  auto rhs = mfProb.rhs;

  mfOp->compute_rhs(*sln, *rhs);
  solver.create_problem(mfProb);
  solver.set_max_iteration_count(1000);
  solver.create_solver();

  std::cout << "problem set, generating output mesh ... : " << std::endl;

  stk::io::StkMeshIoBroker io(bulk.parallel());
  io.set_bulk_data(bulk);
  auto fileId = io.create_output_mesh("conduction.e", stk::io::WRITE_RESULTS);
  io.add_field(fileId, qField);
  io.add_field(fileId, qTmpField);
  io.process_output_request(fileId, 0.0);

  std::cout << "output mesh generated, creating solver... : " << std::endl;

  std::cout << "------------- (presolve)" << std::endl;
//  print_local_data(*solver.prob_->getLHS(), *solver.prob_->getRHS());
  std::cout << "------------- (solve)" << std::endl;
  solver.solve();
  std::cout << "------------- (post solve)" << std::endl;
//  auto solution = solver.prob_->getLHS();
//  print_local_data(*solver.prob_->getLHS(), *solver.prob_->getRHS());
//  std::cout << "------------- (residual)" << std::endl;
//  mfOp->apply(*solution, *resid);
//  resid->update(-1.0, *rhs, 1.0);
//  print_local_data(*solution, *resid);
  std::cout << "------------- (update)" << std::endl;
  update_solution(bulk, meta.universal_part(), entToLid, sln->getLocalView<HostSpace>(), qTmpField, qField);
  std::cout << "linear iterations: " << solver.iteration_count() << std::endl;
  io.process_output_request(fileId, 1.0);
//  std::cout << "finito" << std::endl;
}

TEST_F(TpetraMatrixFreeFixture, check_laplacian_one_cube_element)
{
  constexpr int p = 1;
  set_mesh(1, 1);

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part()))
  {
    for (const auto node : *ib)
    {
      const double qval = coeff(rng);
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), node) = qval;
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateN), node) = qval;
      *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNM1), node) = qval;

      *stk::mesh::field_data(rhocpField, node) = 1.0e0;
      *stk::mesh::field_data(diffusivityField, node) = 1.0e0;
    }
  }
  using interior_operator_type = ConductionInteriorOperator<p>;
  using mfop_type = MFOperator<interior_operator_type, NoOperator>;

  auto entToLid = serial_entity_offset_to_row_lid_map(bulk, meta.universal_part());
  auto mfInterior = ConductionInteriorOperator<p>(bulk, meta.universal_part(), coordField, rhocpField, diffusivityField, entToLid);
  auto mfBdry = NoOperator();
  auto mfOp = make_rcp<mfop_type>(mfInterior, mfBdry, operator_map(bulk, gidField, meta.universal_part(),{},{}));

  auto x = make_rcp<mv_type>(mfOp->getDomainMap(), 1);
  auto xv = x->getLocalView<HostSpace>();

  std::array<double,8> q;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part()))
  {
    for (const auto node : *ib)
    {
      const int lid = entToLid(node.local_offset());
      xv(lid,0) = coeff(rng);
      q[lid] = (*stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1),node)) + xv(lid,0);
    }
  }


  for (int k = 0; k < xv.extent_int(0); ++k) {
    xv(k,0) = coeff(rng);
    q[entToLid[k+1]] = (*stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), stk::mesh::Entity(k+1))) + xv(k,0);

  }  auto y = make_rcp<mv_type>(mfOp->getRangeMap(), 1);
  y->putScalar(0.0);

  constexpr double lhs[8][8] = {
      {1.5, -0.5, -0.5, 0, -0.5, 0, 0, 0},
      {-0.5, 1.5, 0, -0.5, 0, -0.5, 0, 0},
      {-0.5, 0, 1.5, -0.5, 0, 0, -0.5, 0},
      {0, -0.5, -0.5, 1.5, 0, 0, 0, -0.5},
      {-0.5, 0, 0, 0, 1.5, -0.5, -0.5, 0},
      {0, -0.5, 0, 0, -0.5, 1.5, 0, -0.5},
      {0, 0, -0.5, 0, -0.5, 0, 1.5, -0.5},
      {0, 0, 0, -0.5, 0, -0.5, -0.5, 1.5}
  };


  std::array<double,8> result;
  for (int j = 0; j < 8; ++j) {
    result[j] = 0;
    for (int i = 0; i < 8; ++i) {
      result[j] += lhs[j][i]*q[i];
    }
  }

  mfOp->compute_rhs(*x,*y);
  auto yv = y->getLocalView<HostSpace>();
  for (int j = 0; j < 8; ++j) {
    EXPECT_DOUBLE_EQ(yv(j,0), result[j]);
  }
}


}}
