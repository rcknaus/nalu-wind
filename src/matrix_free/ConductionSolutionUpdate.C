#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionJacobiPreconditioner.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/Coefficients.h"

#include <Kokkos_Parallel.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <mpi.h>
#include "stk_ngp/NgpProfilingBlock.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionSolutionUpdate<p>::ConductionSolutionUpdate(
  Teuchos::ParameterList params,
  const ngp::Mesh& mesh_in,
  const stk::mesh::Field<stk::mesh::EntityId>& stk_gid,
  const stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
    tpetra_gid,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet,
  stk::mesh::Selector flux)
  : stk_entity_to_tpetra_index_(
      entity_to_row_lid_mapping(mesh_in, stk_gid, tpetra_gid, active_in)),
    tpetra_index_to_stk_mesh_index_(
      row_lid_to_mesh_index_mapping(mesh_in, stk_entity_to_tpetra_index_)),
    offsets_(
      create_offset_map<p>(mesh_in, active_in, stk_entity_to_tpetra_index_)),
    dirichlet_bc_offsets_(
      simd_node_offsets(mesh_in, dirichlet, stk_entity_to_tpetra_index_)),
    flux_bc_offsets_(
      face_offsets<p>(mesh_in, flux, stk_entity_to_tpetra_index_)),
    exporter_(
      owned_and_shared_row_map(mesh_in, stk_gid, tpetra_gid, active_in),
      owned_row_map(mesh_in, stk_gid, active_in)),
    resid_op_(offsets_, exporter_),
    lin_op_(offsets_, exporter_),
    jacobi_preconditioner_(
      offsets_,
      exporter_,
      params.isParameter("Number of Sweeps")
        ? params.get<int>("Number of Sweeps")
        : 1),
    linear_solver_(
      lin_op_, ConductionLinearizedResidualOperator<p>::num_vectors, params),
    owned_and_shared_mv_(exporter_.getSourceMap(), 1)
{
}

template <int p>
void
ConductionSolutionUpdate<p>::compute_preconditioner(
  double gamma, LinearizedResidualFields<p> coeffs)
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_preconditioner");
  linear_solver_.set_preconditioner(jacobi_preconditioner_);
  jacobi_preconditioner_.set_dirichlet_nodes(dirichlet_bc_offsets_);
  jacobi_preconditioner_.set_coefficients(gamma, coeffs);
  jacobi_preconditioner_.compute_diagonal();
  jacobi_preconditioner_.set_linear_operator(Teuchos::rcpFromRef(lin_op_));
}

template <int p>
void
ConductionSolutionUpdate<p>::compute_residual(
  Kokkos::Array<double, 3> gammas,
  ResidualFields<p> fields,
  BCFields dirichlet_bc_fields,
  BCFluxFields<p> flux_bc_fields)
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_residual");
  resid_op_.set_fields(gammas, fields);
  resid_op_.set_bc_fields(
    dirichlet_bc_offsets_, dirichlet_bc_fields.qp1, dirichlet_bc_fields.qbc);
  resid_op_.set_flux_fields(
    flux_bc_offsets_, flux_bc_fields.exposed_areas, flux_bc_fields.flux);
  resid_op_.compute(linear_solver_.rhs());
}
namespace {

void
copy_tpetra_solution_vector_to_stk_field(
  const_mesh_index_row_view_type lide,
  const typename Tpetra::MultiVector<>::dual_view_type::t_dev delta_view,
  ngp::Field<double>& delta_stk_field)
{
  ngp::ProfilingBlock pf("copy_tpetra_solution_vector_to_stk_field");
  Kokkos::parallel_for(
    delta_view.extent_int(0), KOKKOS_LAMBDA(int k) {
      delta_stk_field.get(lide(k), 0) = delta_view(k, 0);
    });
  delta_stk_field.modify_on_device();
}

} // namespace
template <int p>
void
ConductionSolutionUpdate<p>::compute_delta(
  double gamma, LinearizedResidualFields<p> coeffs, ngp::Field<double>& delta)
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::compute_delta");

  lin_op_.set_coefficients(gamma, coeffs);
  lin_op_.set_dirichlet_nodes(dirichlet_bc_offsets_);
  linear_solver_.solve();
  owned_and_shared_mv_.doImport(
    linear_solver_.lhs(), exporter_, Tpetra::INSERT);
  copy_tpetra_solution_vector_to_stk_field(
    tpetra_index_to_stk_mesh_index_, owned_and_shared_mv_.getLocalViewDevice(),
    delta);
  exec_space().fence();
}

template <int p>
double
ConductionSolutionUpdate<p>::residual_norm() const
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::residual_norm");
  return linear_solver_.nonlinear_residual();
}

template <int p>
double
ConductionSolutionUpdate<p>::final_linear_norm() const
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::final_linear_norm");
  return linear_solver_.final_linear_norm();
}

template <int p>
int
ConductionSolutionUpdate<p>::num_iterations() const
{
  ngp::ProfilingBlock pf("ConductionSolutionUpdate<p>::num_iterations");
  return linear_solver_.num_iterations();
}

INSTANTIATE_POLYSTRUCT(ConductionSolutionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
