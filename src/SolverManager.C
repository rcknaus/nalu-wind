/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "SolverManager.h"
#include "MatrixFreeTraits.h"

#include "PNGSolver.h"
#include "MomentumSolver.h
#include "ContinuitySolver.h"

#include <type_traits>

namespace sierra { namespace nalu {

template <template <int> class Solver> SolverManager<Solver>::SolverManager(int p) : poly_(p) {}
template <template <int> class Solver> SolverManager<Solver>::~SolverManager() = default;

template <template <int> class Solver> void SolverManager<Solver>::create()
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


template <template <int> class Solver> void SolverManager<Solver>::initialize()
{
  switch(poly_)
  {
    case POLY1: solv1_.initialize(); break;
    case POLY2: solv2_.initialize(); break;
    case POLY3: solv3_.initialize(); break;
    case POLY4: solv4_.initialize(); break;
    default: break;
  }
}
template <template <int> class Solver> void SolverManager<Solver>::assemble_and_solve(elem_view::scs_scalar_view mdot)
{
  switch(poly_)
  {
    case POLY1: solv1_.assemble_and_solve(mdot); break;
    case POLY2: solv2_.assemble_and_solve(mdot); break;
    case POLY3: solv3_.assemble_and_solve(mdot); break;
    case POLY4: solv4_.assemble_and_solve(mdot); break;
    default: break;
  }
}

template <template <int> class Solver> void SolverManager<Solver>::assemble_and_solve()
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

template class SolverManager<MomentumSolver<POLY1>>;
template class SolverManager<MomentumSolver<POLY2>>;
template class SolverManager<MomentumSolver<POLY3>>;
template class SolverManager<MomentumSolver<POLY4>>;

template class SolverManager<ContinuitySolver<POLY1>>;
template class SolverManager<ContinuitySolver<POLY2>>;
template class SolverManager<ContinuitySolver<POLY3>>;
template class SolverManager<ContinuitySolver<POLY4>>;

template class SolverManager<PNGSolver<POLY1>>;
template class SolverManager<PNGSolver<POLY2>>;
template class SolverManager<PNGSolver<POLY3>>;
template class SolverManager<PNGSolver<POLY4>>;

} // namespace nalu
} // namespace Sierra

#endif

