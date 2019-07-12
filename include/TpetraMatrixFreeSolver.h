/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TpetraMatrixFreeSolver_h
#define TpetraMatrixFreeSolver_h

#include "Teuchos_RCP.hpp"
#include "MatrixFreeTypes.h"
#include "Tpetra_CrsMatrix.hpp"
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>

#include "nalu_make_unique.h"

namespace Teuchos { class ParameterList; }
namespace Belos { template <typename, typename, typename> class PseudoBlockGmresSolMgr; }
namespace Belos { template <typename, typename, typename, bool> class PseudoBlockCGSolMgr; }
namespace Belos { template <typename, typename, typename> class BlockGmresSolMgr; }
namespace Belos { template <typename, typename, typename> class LinearProblem; }
namespace MueLu { template <typename, typename, typename, typename> class TpetraOperator; }


namespace sierra { namespace nalu {

//using krylov_solver_type = Belos::PseudoBlockCGSolMgr<double, mv_type, operator_type, true>;
using krylov_solver_type = Belos::PseudoBlockGmresSolMgr<double, mv_type, operator_type>;
//using krylov_solver_type = Belos::BlockGmresSolMgr<double, mv_type, operator_type>;
//using krylov_solver_factory = Belos::SolverManager<double, mv_type, operator_type>;
using matrix_type = Tpetra::CrsMatrix<double, int, long>;
using node_type = typename matrix_type::node_type;

using muelu_preconditioner_type = MueLu::TpetraOperator<double, local_ordinal_type, global_ordinal_type, node_type>;
using ifpack2_preconditioner_type = Ifpack2::Preconditioner<double, local_ordinal_type, global_ordinal_type>;

struct MatrixFreeProblem
{
  MatrixFreeProblem(Teuchos::RCP<operator_type> in_op, int ndim)
  : op(in_op),
    sln(make_rcp<mv_type>(in_op->getDomainMap(), ndim)),
    rhs(make_rcp<mv_type>(in_op->getRangeMap(), ndim))
  {
    sln->putScalar(0.);
    rhs->putScalar(0.);
  }

  double residual_norm();


  Teuchos::RCP<Belos::LinearProblem<double, mv_type, operator_type>> make_linear_problem();

  Teuchos::RCP<operator_type> op;
  Teuchos::RCP<mv_type> sln;
  Teuchos::RCP<mv_type> rhs;
};

class TpetraMatrixFreeSolver
{
public:
  explicit TpetraMatrixFreeSolver(int ndim);
  void set_params(Teuchos::RCP<Teuchos::ParameterList>);
  void set_preconditioner_params(Teuchos::RCP<Teuchos::ParameterList>);

  void set_max_iteration_count(int);
  void set_tolerance(double);

  void create_problem(MatrixFreeProblem& mfProb);

  Belos::LinearProblem<double, mv_type, operator_type>& problem() { return *prob_; }

  void create_solver();
  void create_ifpack2_preconditioner(Teuchos::RCP<matrix_type> A);
  void create_muelu_preconditioner(Teuchos::RCP<matrix_type> A, Teuchos::RCP<mv_type> coords);

  double scaled_linear_residual();
  int iteration_count();

  void solve();
//private:

  const int ndim_{1};
  Teuchos::RCP<mv_type> resid_{Teuchos::null};
  Teuchos::RCP<Teuchos::ParameterList> solvParams_{Teuchos::null};
  Teuchos::RCP<Teuchos::ParameterList> precondParams_{Teuchos::null};

  Teuchos::RCP<ifpack2_preconditioner_type> ifpack2Precond_{Teuchos::null};
  Teuchos::RCP<muelu_preconditioner_type> mueluPrecond_{Teuchos::null};

  Teuchos::RCP<Belos::LinearProblem<double, mv_type, operator_type>> prob_{Teuchos::null};
  Teuchos::RCP<krylov_solver_type> solv_{Teuchos::null};
};

} // namespace nalu
} // namespace Sierra

#endif

