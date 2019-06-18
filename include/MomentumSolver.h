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

namespace sierra { namespace nalu {

template <int p> class MomentumSolver
{
public:
  using mfop_type = MFOperatorParallel<MomentumInteriorOperator<p>, NoOperator>;

  static constexpr bool precondition = true;


  MomentumSolver(MomentumEquationSystem& eqSys) : eqSys_(eqSys) {}
  ~MomentumSolver() = default;

  void create()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    auto entToLid = tpetralinsys.entityToLID_;

    mfInterior_ = make_rcp<MomentumInteriorOperator<p>>(bulk, selector, *eqSys_.coordinates_, entToLid);
    mfBdry_ = make_rcp<NoOperator>();

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
      mfDiag_ = make_rcp<MomentumInteriorDiagonal<p>>(bulk, selector, tpetralinsys, *eqSys_.coordinates_);
      solver_->create_ifpack2_preconditioner(tpetralinsys.ownedMatrix_);
    }

    solver_->create_problem(*mfProb_);
    solver_->set_tolerance(1.0e-4);
    solver_->set_max_iteration_count(50);
    solver_->create_solver();
  }

  void initialize()
  {
    auto& realm = eqSys_.realm_;
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    mfInterior_->initialize(bulk, selector);
  }

  void assemble(ko::scs_scalar_view<p> mdot)
  {
    auto& realm = eqSys_.realm_;
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();

    mfInterior_->update_element_views(bulk, selector);

    mfInterior_->set_gamma({{
      realm.get_gamma1()/realm.get_time_step(),
      realm.get_gamma2()/realm.get_time_step(),
      realm.get_gamma3()/realm.get_time_step()
    }});

    mfInterior_->set_mdot(mdot);
    mfOp_->compute_rhs(*mfProb_->rhs);

    if (precondition) {
      auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
      mfDiag_->set_gamma(realm.get_gamma1()/realm.get_time_step());
      mfDiag_->set_mdot(mfInterior_->mdot_);
      tpetralinsys.zeroSystem();
      mfDiag_->compute_diagonal();
      tpetralinsys.loadComplete();
    }
  }

  void solve()
  {
    solver_->solve();
    std::cout << "momentum it count: " << solver_->iteration_count() << std::endl;
  }

//  void banner()
//  {
//    auto iters = solver_->iteration_count()
//    auto
//
//  }


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
  MomentumEquationSystem& eqSys_;
  Teuchos::RCP<MomentumInteriorOperator<p>> mfInterior_;
  Teuchos::RCP<NoOperator> mfBdry_;
  Teuchos::RCP<MomentumInteriorDiagonal<p>> mfDiag_;
  Teuchos::RCP<MatrixFreeProblem> mfProb_;
  Teuchos::RCP<mfop_type> mfOp_;
  Teuchos::RCP<TpetraMatrixFreeSolver> solver_;
};


} // namespace nalu
} // namespace Sierra

#endif

