#include <math.h>
#include <stdlib.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <algorithm>
#include <random>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldState.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include "matrix_free/ConductionOperator.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/ConductionFields.h"
#include "StkConductionFixture.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/make_rcp.hpp"
#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class OperatorFixture : public ConductionFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);

  OperatorFixture()
    : ConductionFixture(nx, scale),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part())),
      elid(entity_to_row_lid_mapping(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      offsets(create_offset_map<order>(mesh, meta.universal_part(), elid)),
      owned_map(owned_row_map(mesh, gid_field, meta.universal_part())),
      owned_and_shared_map(owned_and_shared_row_map(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      exporter(owned_and_shared_map, owned_map),
      coordinate_ordinal(coordinate_field().mesh_meta_data_ordinal()),
      residual_ordinals(conduction_field_ordinals(meta)),
      coefficient_ordinals(conduction_coefficient_ordinals(meta)),
      lhs(owned_map, 1),
      rhs(owned_map, 1)
  {
    lhs.putScalar(0.);
    rhs.putScalar(0.);
  }

  elem_mesh_index_view<order> conn;
  entity_row_view_type elid;
  elem_offset_view<order> offsets;
  node_offset_view dirichlet_bc_nodes{"dirichlet_bc_nodes_empty", 0};

  Teuchos::RCP<const Tpetra::Map<>> owned_map;
  Teuchos::RCP<const Tpetra::Map<>> owned_and_shared_map;
  Tpetra::Export<> exporter;

  int coordinate_ordinal{-1};
  Kokkos::Array<int, conduction_info::num_physics_fields> residual_ordinals;
  Kokkos::Array<int, conduction_info::num_coefficient_fields>
    coefficient_ordinals;

  Tpetra::MultiVector<> lhs;
  Tpetra::MultiVector<> rhs;

  static constexpr int nx = 3;
  static constexpr double scale = M_PI;
  static constexpr Kokkos::Array<double, 3> gammas = {{+1, -1, 0}};
};

TEST_F(OperatorFixture, residual_operator_zero_for_constant_data)
{
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = 1.0;
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        1.0;
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = 1.0;
      *stk::mesh::field_data(alpha_field, node) = 1.0;
      *stk::mesh::field_data(lambda_field, node) = 1.0;
    }
  }

  auto fields = gather_required_conduction_fields<order>(
    conn, fm, coordinate_ordinal, residual_ordinals);
  ConductionResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields({{+1, -1, 0}}, fields);
  resid_op.compute(rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();

  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.0e-14);
  }
}

TEST_F(OperatorFixture, residual_operator_not_zero_for_nonconstant_data)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(1.0, 2.0);
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = coeff(rng);
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        coeff(rng);
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = coeff(rng);
      *stk::mesh::field_data(alpha_field, node) = coeff(rng);
      *stk::mesh::field_data(lambda_field, node) = coeff(rng);
    }
  }
  stk::mesh::copy_owned_to_shared(
    mesh.get_bulk_on_host(), {&alpha_field, &lambda_field});

  auto fields = gather_required_conduction_fields<order>(
    conn, fm, coordinate_ordinal, residual_ordinals);

  ConductionResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields({{+1, -1, 0}}, fields);
  resid_op.compute(rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();
  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}
//
TEST_F(OperatorFixture, linearized_residual_operator_zero_for_constant_data)
{
  auto fields = gather_required_conduction_fields<order>(
    conn, fm, coordinate_ordinal, residual_ordinals);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  ConductionLinearizedResidualOperator<order> cond_op(offsets, exporter);
  cond_op.set_coefficients(gammas[0], coefficient_fields);

  lhs.putScalar(0.);
  cond_op.apply(lhs, rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.e-14);
  }
}

TEST_F(
  OperatorFixture, linearized_residual_operator_not_zero_for_nonconstant_data)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(1.0, 2.0);
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(alpha_field, node) = coeff(rng);
      *stk::mesh::field_data(lambda_field, node) = coeff(rng);
    }
  }
  stk::mesh::copy_owned_to_shared(
    mesh.get_bulk_on_host(), {&alpha_field, &lambda_field});

  auto fields = gather_required_conduction_fields<order>(
    conn, fm, coordinate_ordinal, residual_ordinals);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  ConductionLinearizedResidualOperator<order> cond_op(offsets, exporter);
  cond_op.set_coefficients(gammas[0], coefficient_fields);

  lhs.randomize(-1, +1);
  lhs.sync_device();
  cond_op.apply(lhs, rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();

  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
