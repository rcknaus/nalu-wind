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
  using mfop_type = MFOperatorParallel<ProjectedNodalGradientInteriorOperator<p>, NoOperator>;

  static constexpr bool precondition = true;

  PNGSolver(ProjectedNodalGradientEquationSystem& eqSys) : eqSys_(eqSys) {}

  void create()
  {
    auto& realm = eqSys_.realm_;
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(eqSys_.linsys_);
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    auto& meta = realm.meta_data();
    auto entToLid = tpetralinsys.entityToLID_;

    auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK,  "coordinates");
    mfInterior_ = make_rcp<ProjectedNodalGradientInteriorOperator<p>>(bulk, selector, coordField, entToLid);
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
      mfMass_ = make_rcp<MassInteriorDiagonal<p>>(bulk, selector, tpetralinsys, coordField);
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
    auto selector  = stk::mesh::selectUnion(realm.interiorPartVec_);
    auto& bulk = realm.bulk_data();
    const auto& qField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK,
      "pressure");

    const auto& dqdxField = *bulk.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK,
      "dpdx");
    mfInterior_->initialize(bulk, selector, qField, dqdxField);
    mfOp_->compute_rhs(*mfProb_->rhs);
  }

  void solve()
  {
    solver_->solve();
    std::cout << "png it count: " << solver_->iteration_count() << std::endl;

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
  ProjectedNodalGradientEquationSystem& eqSys_;
  Teuchos::RCP<ProjectedNodalGradientInteriorOperator<MF::p>> mfInterior_;
  Teuchos::RCP<NoOperator> mfBdry_;
  Teuchos::RCP<MassInteriorDiagonal<MF::p>> mfMass_;
  Teuchos::RCP<MatrixFreeProblem> mfProb_;
  Teuchos::RCP<mfop_type> mfOp_;
  Teuchos::RCP<TpetraMatrixFreeSolver> solver_;
};

} // namespace nalu
} // namespace Sierra

#endif

