#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "StkConductionFixture.h"

#include <math.h>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "BelosCGIteration.hpp"
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosMultiVecTraits.hpp"
#include "BelosOperatorTraits.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosStatusTestGenResNorm.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosTypes.hpp"
#include "Kokkos_Core.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace test_belos_solver {
static constexpr Kokkos::Array<double, 3> gammas{{+1, -1, 0}};
}

class SolverFixture : public ::ConductionFixture
{
protected:
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;

  SolverFixture()
    : ConductionFixture(nx, scale),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part())),
      elid(entity_to_row_lid_mapping(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      offsets(create_offset_map<order>(mesh, meta.universal_part(), elid)),
      owned_map(owned_row_map(mesh, gid_field, meta.universal_part())),
      owned_and_shared_map(owned_and_shared_row_map(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      exporter(owned_and_shared_map, owned_map),
      resid_op(offsets, exporter),
      lin_op(offsets, exporter)
  {
    auto& coordField = coordinate_field();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coordField, node);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = 1.0;
        *stk::mesh::field_data(lambda_field, node) = 1.0;
      }
    }

    const auto coordinate_ordinal = coordinate_field().mesh_meta_data_ordinal();
    fields = gather_required_conduction_fields<order>(
      conn, fm, coordinate_ordinal, conduction_field_ordinals(meta));
    coefficient_fields.volume_metric = fields.volume_metric;
    coefficient_fields.diffusion_metric = fields.diffusion_metric;
  }
  const elem_mesh_index_view<order> conn;
  const const_entity_row_view_type elid;
  const elem_offset_view<order> offsets;
  const node_offset_view dirichlet_bc_offsets{"empty_dirichlet", 0};
  const Teuchos::RCP<const Tpetra::Map<>> owned_map;
  const Teuchos::RCP<const Tpetra::Map<>> owned_and_shared_map;

  const Tpetra::Export<> exporter;

  ConductionResidualOperator<order> resid_op;
  ConductionLinearizedResidualOperator<order> lin_op;

  ResidualFields<order> fields;
  LinearizedResidualFields<order> coefficient_fields;
};

TEST_F(SolverFixture, solve_zero_rhs)
{
  auto list = Teuchos::ParameterList{};
  MatrixFreeSolver solver(lin_op, 1, list);
  lin_op.set_coefficients(test_belos_solver::gammas[0], coefficient_fields);

  solver.rhs().putScalar(0.);
  solver.solve();
  ASSERT_EQ(solver.num_iterations(), 0);
}

TEST_F(SolverFixture, solve_harmonic)
{
  auto list = Teuchos::ParameterList{};
  MatrixFreeSolver solver(lin_op, 1, list);
  resid_op.set_fields(test_belos_solver::gammas, fields);
  resid_op.compute(solver.rhs());
  lin_op.set_coefficients(test_belos_solver::gammas[0], coefficient_fields);
  solver.solve();
  ASSERT_TRUE(solver.num_iterations() > 1 && solver.num_iterations() < 1000);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
