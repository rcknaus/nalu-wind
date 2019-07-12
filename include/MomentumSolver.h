/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumSolver_h
#define MomentumSolver_h

#include "Realm.h"
#include "LowMachEquationSystem.h"
#include "MomentumInteriorOperator.h"
#include "TpetraLinearSystem.h"
#include "MatrixFreeOperator.h"
#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeTypes.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include "ConductionInteriorOperator.h"
#include "ContinuityInteriorOperator.h"
#include "SparsifiedLaplacian.h"
#include "MomentumDiagonal.h"
#include "DirichletVectorOperator.h"


namespace sierra { namespace nalu {

template <int p> class MomentumSolver
{
public:
  using mfop_type = MFOperatorParallel<MomentumInteriorOperator<p>, DirichletVectorOperator>;

  static constexpr bool precondition = true;

  MomentumSolver(MomentumEquationSystem& eqSys) : eqSys_(eqSys) {}
  ~MomentumSolver() = default;

  void create()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto interiorSelector = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto wallSelector = stk::mesh::selectUnion(realm.wallPartVec_);

    auto& bulk = realm.bulk_data();
    auto entToLid = tpetralinsys.entityToLID_;

    mfInterior_ = make_rcp<MomentumInteriorOperator<p>>(bulk, interiorSelector, *eqSys_.coordinates_, entToLid);
    mfBdry_ = make_rcp<DirichletVectorOperator>(bulk, wallSelector, entToLid);

    mfOp_ = make_rcp<mfop_type>(
      *mfInterior_,
      *mfBdry_,
      tpetralinsys.maxOwnedRowId_,
      tpetralinsys.maxSharedNotOwnedRowId_,
      tpetralinsys.ownedRowsMap_,
      tpetralinsys.sharedNotOwnedRowsMap_,
      tpetralinsys.sharedAndOwnedRowsMap_,
      3
    );
    mfProb_ = make_rcp<MatrixFreeProblem>(mfOp_, 3);
    auto sln = mfProb_->sln;
    auto rhs = mfProb_->rhs;
    solver_ = make_rcp<TpetraMatrixFreeSolver>(3);
    if (precondition) {
      mfDiag_ = make_rcp<MomentumInteriorDiagonal<p>>(bulk, interiorSelector, tpetralinsys, *eqSys_.coordinates_);
      solver_->create_ifpack2_preconditioner(tpetralinsys.ownedMatrix_);
    }

    solver_->create_problem(*mfProb_);
    solver_->set_tolerance(1.0e-4);
    solver_->set_max_iteration_count(100);
    solver_->create_solver();
  }

  void initialize()
  {
    auto& realm = eqSys_.realm_;
     auto interiorSelector  = stk::mesh::selectUnion(realm.interiorPartVec_);
     auto wallSelector  = stk::mesh::selectUnion(realm.wallPartVec_);
     auto& bulk = realm.bulk_data();
     auto& meta = realm.meta_data();

    mfInterior_->initialize(bulk, interiorSelector);

    const auto& solnField = eqSys_.velocity_->field_of_state(stk::mesh::StateNP1);
    const auto& bcField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_bc");
    mfBdry_->initialize(bulk, wallSelector, solnField, bcField);
  }

  void assemble(elem_view::scs_scalar_view<p> mdot)
  {
    auto& realm = eqSys_.realm_;
    auto interiorSelector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto wallSelector  = stk::mesh::selectUnion(realm.wallPartVec_);
    auto& bulk = realm.bulk_data();
    auto& meta = realm.meta_data();

    mfInterior_->update_element_views(bulk, interiorSelector);

    mfInterior_->set_gamma({{
      realm.get_gamma1()/realm.get_time_step(),
      realm.get_gamma2()/realm.get_time_step(),
      realm.get_gamma3()/realm.get_time_step()
    }});
    mfInterior_->set_mdot(mdot);

    auto& solnField = eqSys_.velocity_->field_of_state(stk::mesh::StateNP1);
    auto& bcField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity_bc");
    mfBdry_->initialize(bulk, wallSelector, solnField, bcField);

    mfOp_->compute_rhs(*mfProb_->rhs);
    if (precondition) {
      auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
      mfDiag_->set_gamma(realm.get_gamma1()/realm.get_time_step());
      mfDiag_->set_mdot(mfInterior_->mdot_);
      tpetralinsys.zeroSystem();
      mfDiag_->compute_diagonal();
      tpetralinsys.applyDirichletBCs(&solnField, &bcField, realm.wallPartVec_, 0u, 1u);
      tpetralinsys.loadComplete();
    }
  }

  void banner()
  {
    const auto iters = solver_->iteration_count();
    const auto linResid = solver_->scaled_linear_residual();
    const auto norm = mfProb_->residual_norm() * eqSys_.realm_.l2Scaling_;
    if (eqSys_.firstTimeStepSolve_) {
      firstResid = (norm > std::numeric_limits<double>::epsilon()) ? norm : 1;
      eqSys_.firstTimeStepSolve_ = false;
    }
    eqSys_.output_banner(iters, linResid, norm, norm / firstResid);
  }

  void solve()
  {
    solver_->solve();
    banner();
  }


  void update_solution()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    auto entToLid = tpetralinsys.entityToLID_;

    write_solution_to_field(bulk, selector, entToLid, mfProb_->sln->getLocalView<HostSpace>(), *eqSys_.uTmp_);
    if (realm.hasPeriodic_) {
      realm.periodic_delta_solution_update(eqSys_.uTmp_, 3);
    }
  }

  void assemble_and_solve(elem_view::scs_scalar_view<p> mdot)
  {
    double timeA = NaluEnv::self().nalu_time();
    assemble(mdot);
    double timeB = NaluEnv::self().nalu_time();
    eqSys_.timerAssemble_ += (timeB - timeA);

    timeA = NaluEnv::self().nalu_time();
    solve();
    timeB = NaluEnv::self().nalu_time();
    eqSys_.timerSolve_ += (timeB - timeA);

    timeA = NaluEnv::self().nalu_time();
    update_solution();
    timeB = NaluEnv::self().nalu_time();
    eqSys_.timerAssemble_ += (timeB - timeA);
  }


  MomentumEquationSystem& eqSys_;
  Teuchos::RCP<MomentumInteriorOperator<p>> mfInterior_;
  Teuchos::RCP<DirichletVectorOperator> mfBdry_;
  Teuchos::RCP<MomentumInteriorDiagonal<p>> mfDiag_;
  Teuchos::RCP<MatrixFreeProblem> mfProb_;
  Teuchos::RCP<mfop_type> mfOp_;
  Teuchos::RCP<TpetraMatrixFreeSolver> solver_;


  double firstResid{0};
};


} // namespace nalu
} // namespace Sierra

#endif

