/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef PNGSolver_h
#define PNGSolver_h

#include "Realm.h"
#include "ProjectedNodalGradientEquationSystem.h"

#include "ProjectedNodalGradientInteriorOperator.h"
#include "TpetraLinearSystem.h"
#include "MatrixFreeOperator.h"
#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeTypes.h"
#include "NoBoundaryOperator.h"
#include "PNGBoundaryOperator.h"
#include "ProjectedNodalGradientInteriorOperator.h"
#include "MassDiagonal.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>

namespace sierra { namespace nalu {


template <int p> class PNGSolver {
public:
  using mfop_type = MFOperatorParallel<ProjectedNodalGradientInteriorOperator<p>, PNGBoundaryOperator<p>>;

  static constexpr bool precondition = true;

  PNGSolver(ProjectedNodalGradientEquationSystem& eqSys) : eqSys_(eqSys) {}

  void create()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto interiorSelector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto wallSelector  = stk::mesh::selectUnion(realm.wallPartVec_);

    auto& bulk = realm.bulk_data();
    auto& meta = realm.meta_data();
    auto entToLid = tpetralinsys.entityToLID_;

    auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK,  "coordinates");
    mfInterior_ = make_rcp<ProjectedNodalGradientInteriorOperator<p>>(bulk, interiorSelector, coordField, entToLid);
    mfBdry_ = make_rcp<PNGBoundaryOperator<p>>(bulk, wallSelector, coordField, entToLid);

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
      mfMass_ = make_rcp<MassInteriorDiagonal<p>>(bulk, interiorSelector, tpetralinsys, coordField);
      tpetralinsys.zeroSystem();
      mfMass_->compute_diagonal();
      tpetralinsys.loadComplete();
      solver_->create_ifpack2_preconditioner(tpetralinsys.ownedMatrix_);
    }
    solver_->create_problem(*mfProb_);
    solver_->set_tolerance(1.0e-4);
    solver_->set_max_iteration_count(100);
    solver_->create_solver();
  }

  void assemble()
  {
    auto& realm = eqSys_.realm_;
    auto interiorSelector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto wallSelector  = stk::mesh::selectUnion(realm.wallPartVec_);

    auto& bulk = realm.bulk_data();
    const auto& qField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK,
      "pressure");

    const auto& dqdxField = *bulk.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK,
      "dpdx");
    mfInterior_->initialize(bulk, interiorSelector, qField, dqdxField);

    const auto& qbcField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK,
      "pressure");

    mfBdry_->initialize(bulk, wallSelector, qbcField);
    mfOp_->compute_rhs(*mfProb_->rhs);
  }

  void banner()
  {
    const auto iters = solver_->iteration_count();
    const auto linResid = solver_->scaled_linear_residual();
    const auto norm = mfProb_->residual_norm() * eqSys_.realm_.l2Scaling_;
    if (eqSys_.firstTimeStepSolve_) {
      firstResid =  norm;
      eqSys_.firstTimeStepSolve_ = false;
    }
    const auto scaledResid = (firstResid > std::numeric_limits<double>::epsilon()) ?
        norm / firstResid : 0;
    eqSys_.output_banner(iters, linResid, norm, scaledResid);
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

    write_solution_to_field(bulk, selector, entToLid, mfProb_->sln->getLocalView<HostSpace>(), *eqSys_.qTmp_);
    if (realm.hasPeriodic_) {
      realm.periodic_delta_solution_update(eqSys_.qTmp_, 3);
    }
  }

  void assemble_and_solve()
  {
    double timeA = NaluEnv::self().nalu_time();
    assemble();
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

  ProjectedNodalGradientEquationSystem& eqSys_;
  Teuchos::RCP<ProjectedNodalGradientInteriorOperator<MF::p>> mfInterior_;
  Teuchos::RCP<PNGBoundaryOperator<p>> mfBdry_;
  Teuchos::RCP<MassInteriorDiagonal<MF::p>> mfMass_;
  Teuchos::RCP<MatrixFreeProblem> mfProb_;
  Teuchos::RCP<mfop_type> mfOp_;
  Teuchos::RCP<TpetraMatrixFreeSolver> solver_;

  double firstResid{0};
};

} // namespace nalu
} // namespace Sierra

#endif

