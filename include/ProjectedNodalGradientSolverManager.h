/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ProjectedNodalGradientSolverManager_h
#define ProjectedNodalGradientSolverManager_h

#include "MatrixFreeTraits.h"
#include "CVFEMTypeDefs.h"
#include "ProjectedNodalGradientEquationSystem.h"

namespace sierra { namespace nalu {

template <int> class PNGSolver;

class ProjectedNodalGradientSolverManager
{
  ProjectedNodalGradientSolverManager(EquationSystem& eqsys, int p);
  ~ProjectedNodalGradientSolverManager();
  void create();
  void assemble_and_solve();
private:
  const int poly_{POLY1};

  PNGSolver<POLY1> solv1_;
  PNGSolver<POLY2> solv2_;
  PNGSolver<POLY3> solv3_;
  PNGSolver<POLY4> solv4_;
};

} // namespace nalu
} // namespace Sierra

#endif

