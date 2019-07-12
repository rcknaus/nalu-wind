/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ProjectedNodalGradientSolverManager.h"
#include "MatrixFreeTraits.h"

#include "PNGSolver.h"

#include <type_traits>

namespace sierra { namespace nalu {

ProjectedNodalGradientSolverManager::ProjectedNodalGradientSolverManager(EquationSystem& eqsys, int p)
: poly_(p),
  solv1_(eqsys),
  solv2_(eqsys),
  solv3_(eqsys),
  solv4_(eqsys)
{}

ProjectedNodalGradientSolverManager::~ProjectedNodalGradientSolverManager() = default;

void ProjectedNodalGradientSolverManager::create()
{
  switch(poly_)
  {
    case POLY1: solv1_.create(); break;
    case POLY2: solv2_.create(); break;
    case POLY3: solv3_.create(); break;
    case POLY4: solv4_.create(); break;
    default: break;
  }
}

void ProjectedNodalGradientSolverManager::assemble_and_solve()
{
  switch(poly_)
  {
    case POLY1: solv1_.assemble_and_solve(); break;
    case POLY2: solv2_.assemble_and_solve(); break;
    case POLY3: solv3_.assemble_and_solve(); break;
    case POLY4: solv4_.assemble_and_solve(); break;
    default: break;
  }
}


} // namespace nalu
} // namespace Sierra

#endif

