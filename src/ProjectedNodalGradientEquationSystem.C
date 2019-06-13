/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ProjectedNodalGradientEquationSystem.h>

#include <AssemblePNGElemSolverAlgorithm.h>
#include <AssemblePNGBoundarySolverAlgorithm.h>
#include <AssemblePNGNonConformalSolverAlgorithm.h>
#include <EquationSystem.h>
#include <EquationSystems.h>
#include <Enums.h>
#include <FieldFunctions.h>
#include <LinearSolvers.h>
#include <LinearSolver.h>
#include <LinearSystem.h>
#include <NaluEnv.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <SolutionOptions.h>
#include <SolverAlgorithmDriver.h>

#include <kernel/KernelBuilder.h>
#include <kernel/ProjectedNodalGradientHOElemKernel.h>

#include "ProjectedNodalGradientInteriorOperator.h"
#include "TpetraLinearSystem.h"
#include "MatrixFreeOperator.h"
#include "TpetraMatrixFreeSolver.h"
#include "MatrixFreeTypes.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include "ConductionInteriorOperator.h"
#include "ProjectedNodalGradientInteriorOperator.h"
#include "SparsifiedLaplacian.h"
#include "MassDiagonal.h"

// user functions
#include <user_functions/SteadyThermalContactAuxFunction.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/SkinMesh.hpp>
#include <stk_mesh/base/Comm.hpp>

// stk_io
#include <stk_io/IossBridge.hpp>
#include <stk_topology/topology.hpp>

// stk_util
#include <stk_util/parallel/Parallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

// overset
#include <overset/UpdateOversetFringeAlgorithmDriver.h>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ProjectedNodalGradientEquationSystem - do some stuff
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ProjectedNodalGradientEquationSystem::ProjectedNodalGradientEquationSystem(
 EquationSystems& eqSystems,
 const EquationType eqType,
 const std::string dofName,
 const std::string deltaName,
 const std::string independentDofName,
 const std::string eqSysName,
 const bool managesSolve)
  : EquationSystem(eqSystems, eqSysName),
    eqType_(eqType),
    dofName_(dofName),
    deltaName_(deltaName),
    independentDofName_(independentDofName),
    eqSysName_(eqSysName),
    managesSolve_(managesSolve),
    dqdx_(NULL),
    qTmp_(NULL)
{
  // extract solver name and solver object
  std::string solverName = realm_.equationSystems_.get_solver_block_name(dofName);
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, eqType_);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);

  // push back EQ to manager
  realm_.push_equation_to_systems(this);
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ProjectedNodalGradientEquationSystem::~ProjectedNodalGradientEquationSystem()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- set_data_map ----------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::set_data_map(
  BoundaryConditionType BC, std::string name)
{
  dataMap_[BC] = name;
}

//--------------------------------------------------------------------------
//-------- get_name_given_bc -----------------------------------------------
//--------------------------------------------------------------------------
std::string
ProjectedNodalGradientEquationSystem::get_name_given_bc(
  BoundaryConditionType BC)
{
  std::map<BoundaryConditionType, std::string>::iterator it;
  it=dataMap_.find(BC);
  if ( it == dataMap_.end() )
    throw std::runtime_error("PNGEqSys::missing BC type specification (developer error)!");
  else
    return it->second;
}

//--------------------------------------------------------------------------
//-------- register_nodal_fields -------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_nodal_fields(
  stk::mesh::Part *part)
{
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  dqdx_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, dofName_));
  stk::mesh::put_field_on_mesh(*dqdx_, *part, nDim, nullptr);

  // delta solution for linear solver
  qTmp_ =  &(meta_data.declare_field<VectorFieldType>(stk::topology::NODE_RANK, deltaName_));
  stk::mesh::put_field_on_mesh(*qTmp_, *part, nDim, nullptr);
}

//--------------------------------------------------------------------------
//-------- register_interior_algorithm -------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_interior_algorithm(
  stk::mesh::Part *part)
{
  // types of algorithms
  const AlgorithmType algType = INTERIOR;

  // solver
  if (!realm_.solutionOptions_->useConsoldiatedPngSolverAlg_) {
    std::map<AlgorithmType, SolverAlgorithm *>::iterator its
    = solverAlgDriver_->solverAlgMap_.find(algType);
    if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
      AssemblePNGElemSolverAlgorithm *theAlg
      = new AssemblePNGElemSolverAlgorithm(realm_, part, this, independentDofName_, dofName_);
      solverAlgDriver_->solverAlgMap_[algType] = theAlg;
    }
    else {
      its->second->partVec_.push_back(part);
    }
  }
  else {
    KernelBuilder kb(*this, *part, solverAlgDriver_->solverAlgorithmMap_, true);

    kb.build_sgl_kernel_automatic<ProjectedNodalGradientHOElemKernel>(
      dofName_ + "_png",
      realm_.bulk_data(), *realm_.solutionOptions_, independentDofName_, dofName_, kb.data_prereqs_HO()
    );
  }
}

//--------------------------------------------------------------------------
//-------- register_wall_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_wall_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const WallBoundaryConditionData &/*wallBCData*/)
{

  const AlgorithmType algType = WALL;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(WALL_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
    AssemblePNGBoundarySolverAlgorithm *theAlg
      = new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_inflow_bc ----------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_inflow_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const InflowBoundaryConditionData &/*inflowBCData*/)
{

  const AlgorithmType algType = INFLOW;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(INFLOW_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
    AssemblePNGBoundarySolverAlgorithm *theAlg
      = new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_open_bc ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_open_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const OpenBoundaryConditionData &/*openBCData*/)
{
  const AlgorithmType algType = OPEN;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(OPEN_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
    AssemblePNGBoundarySolverAlgorithm *theAlg
      = new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_symmetry_bc --------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_symmetry_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/,
  const SymmetryBoundaryConditionData &/*symmetryBCData*/)
{
  const AlgorithmType algType = SYMMETRY;

  // extract the field name for this bc type
  std::string fieldName = get_name_given_bc(SYMMETRY_BC);
  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
    AssemblePNGBoundarySolverAlgorithm *theAlg
      = new AssemblePNGBoundarySolverAlgorithm(realm_, part, this, fieldName);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_non_conformal_bc ---------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_non_conformal_bc(
  stk::mesh::Part *part,
  const stk::topology &/*theTopo*/)
{
  // FIX THIS
  const AlgorithmType algType = NON_CONFORMAL;

  // create lhs/rhs algorithm;
  std::map<AlgorithmType, SolverAlgorithm *>::iterator its =
    solverAlgDriver_->solverAlgMap_.find(algType);
  if ( its == solverAlgDriver_->solverAlgMap_.end() ) {
    AssemblePNGNonConformalSolverAlgorithm *theAlg
      = new AssemblePNGNonConformalSolverAlgorithm(realm_, part, this, independentDofName_, dofName_, realm_.solutionOptions_->ncAlgPngPenalty_);
    solverAlgDriver_->solverAlgMap_[algType] = theAlg;
  }
  else {
    its->second->partVec_.push_back(part);
  }
}

//--------------------------------------------------------------------------
//-------- register_overset_bc ---------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::register_overset_bc()
{
  create_constraint_algorithm(dqdx_);

  int nDim = realm_.meta_data().spatial_dimension();

  // Perform fringe updates before all equation system solves
  UpdateOversetFringeAlgorithmDriver* theAlg = new UpdateOversetFringeAlgorithmDriver(realm_);

  equationSystems_.preIterAlgDriver_.push_back(theAlg);

  theAlg->fields_.push_back(
    std::unique_ptr<OversetFieldData>(new OversetFieldData(dqdx_,1,nDim)));
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::initialize()
{

  if (MF::doMatrixFree) {
    auto& tpetralinsys = *dynamic_cast<TpetraLinearSystem*>(linsys_);
    tpetralinsys.buildSparsifiedElemToNodeGraph(stk::mesh::selectUnion(realm_.interiorPartVec_));
    linsys_->finalizeLinearSystem();
    return;
  }
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- reinitialize_linear_system --------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::reinitialize_linear_system()
{
  // delete linsys; set previously set parameters on linsys
  const bool provideOutput = linsys_->provideOutput_;
  delete linsys_;

  // delete old solver
  const EquationType theEqID = eqType_;
  LinearSolver *theSolver = NULL;
  std::map<EquationType, LinearSolver *>::const_iterator iter
    = realm_.root()->linearSolvers_->solvers_.find(theEqID);
  if (iter != realm_.root()->linearSolvers_->solvers_.end()) {
    theSolver = (*iter).second;
    delete theSolver;
  }

  // create new solver; reset parameters
  std::string solverName = realm_.equationSystems_.get_solver_block_name(dofName_);
  LinearSolver *solver = realm_.root()->linearSolvers_->create_solver(solverName, eqType_);
  linsys_ = LinearSystem::create(realm_, realm_.spatialDimension_, this, solver);
  linsys_->provideOutput_ = provideOutput;

  // initialize
  solverAlgDriver_->initialize_connectivity();
  linsys_->finalizeLinearSystem();
}

//--------------------------------------------------------------------------
//-------- solve_and_update ------------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::solve_and_update()
{
  if ( managesSolve_ )
    solve_and_update_external();
}

class PNGSolver {
public:
  using mfop_type = MFOperatorParallel<ProjectedNodalGradientInteriorOperator<MF::p>, NoOperator>;

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
    mfInterior_ = make_rcp<ProjectedNodalGradientInteriorOperator<MF::p>>(bulk, selector, coordField, entToLid);
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
      mfMass_ = make_rcp<MassInteriorDiagonal<MF::p>>(bulk, selector, tpetralinsys, coordField);
      tpetralinsys.zeroSystem();
      mfMass_->compute_diagonal();
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

    const auto& qField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, eqSys_.dofName_);
    mfInterior_->initialize(bulk, selector, qField, *eqSys_.dqdx_);
    mfOp_->compute_rhs(*mfProb_->rhs);
  }

  void solve()
  {
    solver_->solve();
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
      realm.periodic_delta_solution_update(eqSys_.qTmp_, 1);
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


void
ProjectedNodalGradientEquationSystem::solve_and_update_external()
{
  double timeA, timeB;
  static PNGSolver mfSolver(*this);
  if (isInit_) {
    mfSolver.create();
  }

  for ( int k = 0; k < maxIterations_; ++k ) {

    // projected nodal gradient, load_complete and solve
    if(!MF::doMatrixFree) {
      assemble_and_solve(qTmp_);
    }
    else {

      timeA = NaluEnv::self().nalu_time();
      mfSolver.assemble();
      timeB
      mfSolver().

    }


    // update
    double timeA = NaluEnv::self().nalu_time();
    field_axpby(
      realm_.meta_data(),
      realm_.bulk_data(),
      1.0, *qTmp_,
      1.0, *dqdx_,
      realm_.get_activate_aura());
    double timeB = NaluEnv::self().nalu_time();
    timerAssemble_ += (timeB-timeA);
  }
}

//--------------------------------------------------------------------------
//-------- deactivate_output -----------------------------------------------
//--------------------------------------------------------------------------
void
ProjectedNodalGradientEquationSystem::deactivate_output()
{
  linsys_->provideOutput_ = false;
}

} // namespace nalu
} // namespace Sierra
