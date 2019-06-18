/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeOperator.h"
#include "nalu_make_unique.h"
#include "TpetraLinearSystem.h"

#include <Tpetra_Operator.hpp>
#include <Tpetra_MultiVector.hpp>

#include <BelosLinearProblem.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosStatusTestGenResNorm.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

#include "Tpetra_CrsMatrix.hpp"

namespace sierra { namespace nalu {

Teuchos::RCP<Belos::LinearProblem<double, mv_type, operator_type>> MatrixFreeProblem::make_linear_problem()
{
  return make_rcp<Belos::LinearProblem<double, mv_type, operator_type>>(op, sln, rhs);
}

TpetraMatrixFreeSolver::TpetraMatrixFreeSolver(int ndim) : ndim_(ndim)
{
  auto params = make_rcp<Teuchos::ParameterList>();
  const double tolerance = 1.0e-5;
  params->set("Convergence Tolerance", tolerance);

  params->set("Block Size", ndim);
  params->set("Adaptive Block Size", false);

  const int kspace = 5;
  params->set("Num Blocks", kspace);
  const int max_iterations = 75;
  params->set("Maximum Iterations", max_iterations);
  params->set("Maximum Restarts", std::max(1,max_iterations/kspace + 1));
  std::string orthoType = "ICGS";
  params->set("Orthogonalization", orthoType);
  params->set("Implicit Residual Scaling", "Norm of Preconditioned Initial Residual");

  constexpr bool verbose = false;
  if (verbose) {
    params->set("Output Frequency", 1);
    Teuchos::RCP<std::ostream> belosOutputStream = Teuchos::rcpFromRef(std::cout);
    params->set("Output Stream", belosOutputStream);
    params->set("Verbosity", Belos::Debug + Belos::Warnings + Belos::IterationDetails
      + Belos::OrthoDetails + Belos::FinalSummary
      + Belos::TimingDetails + Belos::StatusTestDetails);
  }
  set_params(params);

  auto precondParams = make_rcp<Teuchos::ParameterList>();
  precondParams->set("relaxation: type", "Jacobi");
  precondParams->set("relaxation: sweeps", 1);
  set_preconditioner_params(precondParams);
}

void TpetraMatrixFreeSolver::set_params(Teuchos::RCP<Teuchos::ParameterList> params) { solvParams_ = params; }
void TpetraMatrixFreeSolver::set_preconditioner_params(Teuchos::RCP<Teuchos::ParameterList> params)
{ precondParams_ = params; }

void TpetraMatrixFreeSolver::create_ifpack2_preconditioner(Teuchos::RCP<matrix_type> matrix)
{
  Ifpack2::Factory factory;
  const std::string preconditionerType ("RELAXATION");
  ifpack2Precond_ = factory.create(preconditionerType, Teuchos::rcp_const_cast<const matrix_type>(matrix), 0);
  ifpack2Precond_->setParameters(*precondParams_);
}

void TpetraMatrixFreeSolver::create_muelu_preconditioner(Teuchos::RCP<matrix_type> matrix, Teuchos::RCP<mv_type> coords)
{
  using lo = local_ordinal_type;
  using go = global_ordinal_type;
  auto params = make_rcp<Teuchos::ParameterList>();

  params->set("xml parameter file", "milestone.xml");

  ThrowRequire(coords != Teuchos::null);
  auto& userParamList = params->sublist("user data");
  userParamList.set("Coordinates", coords);

  ThrowRequire(matrix != Teuchos::null);
  mueluPrecond_ = MueLu::CreateTpetraPreconditioner<double, lo, go, node_type>(
    Teuchos::RCP<Tpetra::Operator<double, lo, go, node_type>>(matrix),  *params);
}

void TpetraMatrixFreeSolver::set_max_iteration_count(int maxIter) {
  solvParams_->set("Maximum Iterations", maxIter);
  if (solv_ != Teuchos::null) {
    solv_->setParameters(solvParams_);
  }
}
void TpetraMatrixFreeSolver::set_tolerance(double tolerance) {
  solvParams_->set("Convergence Tolerance", tolerance);
  if (solv_ != Teuchos::null) {
    solv_->setParameters(solvParams_);
  }
}
void TpetraMatrixFreeSolver::create_problem(MatrixFreeProblem& mfProb) {
  prob_ = mfProb.make_linear_problem();
  if(ifpack2Precond_ != Teuchos::null) {
    prob_->setRightPrec(ifpack2Precond_);
  }
  else if (mueluPrecond_ != Teuchos::null) {
    prob_->setRightPrec(mueluPrecond_);
  }
  const auto& rhs = *prob_->getRHS();
  resid_ = make_rcp<mv_type>(rhs.getMap(), rhs.getNumVectors());
}

void TpetraMatrixFreeSolver::create_solver()
{
  ThrowRequire(prob_ != Teuchos::null && solvParams_ != Teuchos::null);
  solv_ = make_rcp<krylov_solver_type>(prob_, solvParams_);
}

void TpetraMatrixFreeSolver::solve()
{
  ThrowRequire(solv_ != Teuchos::null);

  if (ifpack2Precond_ != Teuchos::null) {
    ifpack2Precond_->compute();
  }
  constexpr bool print = false;
  prob_->setProblem();
  if (print && prob_->getRHS()->getNumVectors() == 3) {
    std::cout << "xlen: " <<prob_->getLHS()->getNumVectors() << ", ylen: "<< prob_->getRHS()->getNumVectors()  << std::endl;
    std::cout << "------------ (presolve)" << std::endl;
    const auto x_view = prob_->getLHS()->getLocalView<HostSpace>();
    const auto y_view =  prob_->getRHS()->getLocalView<HostSpace>();
    ThrowRequire(x_view.extent_int(0) == y_view.extent_int(0));
    for (int k = 0; k < y_view.extent_int(0); ++k) {
      std::cout << k << "(x,y): (" << x_view(k,0) << ", " << y_view(k,0)  << ", " << y_view(k,1) << ", "  << y_view(k,2)<<")" << std::endl;
    }
    std::cout << "------------ (solve)" << std::endl;
  }
  solv_->solve();
  if (print  && prob_->getRHS()->getNumVectors() == 3) {
    std::cout << "------------ (postsolve)"  << std::endl;
    const auto x_view = prob_->getLHS()->getLocalView<HostSpace>();
    const auto y_view =  prob_->getRHS()->getLocalView<HostSpace>();
    for (int k = 0; k < y_view.extent_int(0); ++k) {
      std::cout << k << "(x,y): (" << x_view(k,0) << ", " << y_view(k,0)  << ", " << y_view(k,1) << ", "  << y_view(k,2)<<")" << std::endl;
    }
    std::cout << "------------" << std::endl;
  }
}

int TpetraMatrixFreeSolver::iteration_count() { return solv_->getNumIters(); }

double TpetraMatrixFreeSolver::scaled_linear_residual()
{
  const auto& op = *prob_->getOperator();
  const auto& rhs = *prob_->getRHS();
  const auto& sln = *prob_->getLHS();
  mv_type resid(rhs.getMap(), rhs.getNumVectors());
  op.apply(sln, resid);
  resid.update(-1.0, rhs, 1.0);
  std::vector<double> residual_norms(rhs.getNumVectors());
  resid.norm2(residual_norms);
  double resid_mag = 0;
  for (auto norm : residual_norms) {
    resid_mag += norm*norm;
  }

  std::vector<double> rhs_norms(rhs.getNumVectors());
  double rhs_mag = 0;
  for (auto norm : residual_norms) {
    rhs_mag += norm * norm;
  }
  return resid_mag / rhs_mag;
}


} // namespace nalu
} // namespace Sierra



