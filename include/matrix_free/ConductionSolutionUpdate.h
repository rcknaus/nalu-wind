#ifndef CONDUCTION_SOLUTION_UPDATE_H
#define CONDUCTION_SOLUTION_UPDATE_H

#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionJacobiPreconditioner.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/KokkosFramework.h"
#include <Teuchos_ParameterList.hpp>
#include <Tpetra_MultiVector_decl.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
struct ConductionSolutionUpdate
{
public:
  static constexpr int num_vectors =
    ConductionLinearizedResidualOperator<p>::num_vectors;
  ConductionSolutionUpdate(
    Teuchos::ParameterList params,
    const ngp::Mesh&,
    const stk::mesh::Field<stk::mesh::EntityId>&,
    const stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&,
    stk::mesh::Selector,
    stk::mesh::Selector = {},
    stk::mesh::Selector = {});

  void compute_residual(
    Kokkos::Array<double, 3>,
    ResidualFields<p>,
    BCFields = {},
    BCFluxFields<p> = {});

  void
  compute_delta(double gamma, LinearizedResidualFields<p>, ngp::Field<double>&);
  const MatrixFreeSolver& solver() const { return linear_solver_; }
  void compute_preconditioner(double gamma, LinearizedResidualFields<p>);

  double residual_norm() const;
  double final_linear_norm() const;
  int num_iterations() const;

private:
  const const_entity_row_view_type stk_entity_to_tpetra_index_;
  const const_mesh_index_row_view_type tpetra_index_to_stk_mesh_index_;
  const const_elem_offset_view<p> offsets_;
  const node_offset_view dirichlet_bc_offsets_;
  const face_offset_view<p> flux_bc_offsets_;
  const Tpetra::Export<> exporter_;

  ConductionResidualOperator<p> resid_op_;
  ConductionLinearizedResidualOperator<p> lin_op_;
  JacobiOperator<p> jacobi_preconditioner_;

  MatrixFreeSolver linear_solver_;
  Tpetra::MultiVector<> owned_and_shared_mv_;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
