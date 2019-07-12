/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ProjectedNodalGradientSolverManager_h
#define ProjectedNodalGradientSolverManager_h

#include "EquationSystem.h"
#include "MatrixFreeTraits.h"
#include "CVFEMTypeDefs.h"
#include "PngSolver.h"

namespace sierra { namespace nalu {

template <template <int> class Solver>
class SolverManager
{
  SolverManager(EquationS& eqSys, int p);
  ~SolverManager();
  void create();
  void initialize();
  void assemble_and_solve();
  void assemble_and_solve(elem_view::scs_scalar_view);
private:
  const int poly_{POLY1};

  Solver<POLY1> solv1_;
  Solver<POLY2> solv2_;
  Solver<POLY3> solv3_;
  Solver<POLY4> solv4_;
};

} // namespace nalu
} // namespace Sierra

#endif

