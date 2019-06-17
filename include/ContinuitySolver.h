/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ContinuitySolver_h
#define ContinuitySolver_h

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

template <int p> class ContinuitySolver {
public:
  using mfop_type = MFOperatorParallel<ContinuityInteriorOperator<p>, NoOperator>;

  static constexpr bool precondition = true;

  ContinuitySolver(ContinuityEquationSystem& eqSys) : eqSys_(eqSys) {}

  void create()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    auto& meta = realm.meta_data();
    auto entToLid = tpetralinsys.entityToLID_;

    auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK,  "coordinates");
    mfInterior_ = make_rcp<ContinuityInteriorOperator<p>>(bulk, selector, coordField, entToLid);
    mfBdry_ = make_rcp<NoOperator>();

    mfOp_ = make_rcp<mfop_type>(
      *mfInterior_,
      *mfBdry_,
      tpetralinsys.maxOwnedRowId_,
      tpetralinsys.maxSharedNotOwnedRowId_,
      tpetralinsys.ownedRowsMap_,
      tpetralinsys.sharedNotOwnedRowsMap_,
      tpetralinsys.sharedAndOwnedRowsMap_,
      1
    );
    mfProb_ = make_rcp<MatrixFreeProblem>(mfOp_, 1);
    auto sln = mfProb_->sln;
    auto rhs = mfProb_->rhs;
    solver_ = make_rcp<TpetraMatrixFreeSolver>(1);
    if (precondition) {
      mfSparse_ = make_rcp<SparsifiedLaplacianInterior<p>>(bulk, selector, tpetralinsys, coordField);
      tpetralinsys.zeroSystem();
      mfSparse_->compute_lhs_simd();
      tpetralinsys.loadComplete();
      auto coordMV = make_rcp<mv_type>(sln->getMap(), 3);
      tpetralinsys.copy_stk_to_tpetra(&coordField, coordMV);
      solver_->create_muelu_preconditioner(tpetralinsys.ownedMatrix_, coordMV);
    }
    solver_->create_problem(*mfProb_);
    solver_->set_tolerance(1.0e-4);
    solver_->set_max_iteration_count(100);
    solver_->create_solver();
  }

  void assemble()
  {
    auto& realm = eqSys_.realm_;
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    mfInterior_->initialize(bulk, selector);
    mfInterior_->set_projected_timescale(realm.get_time_step()/realm.get_gamma1());
    compute_mdot();
    mfOp_->compute_rhs(*mfProb_->rhs);
  }

  void compute_mdot()
  {
    auto& realm = eqSys_.realm_;
    mfInterior_->compute_mdot(realm.get_time_step()/realm.get_gamma1());
  }


  void solve()
  {
    solver_->solve();
    std::cout << "continuity it count: " << solver_->iteration_count() << std::endl;
  }

  void update_solution()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    auto entToLid = tpetralinsys.entityToLID_;

    write_solution_to_field(bulk, selector, entToLid, mfProb_->sln->getLocalView<HostSpace>(), *eqSys_.pTmp_);
    if (realm.hasPeriodic_) {
      realm.periodic_delta_solution_update(eqSys_.pTmp_, 1);
    }
  }
  ContinuityEquationSystem& eqSys_;
  Teuchos::RCP<ContinuityInteriorOperator<p>> mfInterior_;
  Teuchos::RCP<NoOperator> mfBdry_;
  Teuchos::RCP<SparsifiedLaplacianInterior<p>> mfSparse_;
  Teuchos::RCP<MatrixFreeProblem> mfProb_;
  Teuchos::RCP<mfop_type> mfOp_;
  Teuchos::RCP<TpetraMatrixFreeSolver> solver_;
};


} // namespace nalu
} // namespace Sierra

#endif

