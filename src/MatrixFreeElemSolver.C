/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <MatrixFreeElemSolver.h>
#include <kernel/ScalarDiffHOElemKernel.h>

namespace sierra{
namespace nalu{

#define INSTANTIATE_MFSOLVER(ClassName,PolyOrder) \
  template class MatrixFreeElemSolver<PolyOrder,ClassName<PolyOrder>>; \

INSTANTIATE_MFSOLVER(ScalarConductionHOElemKernel, 1)

} // namespace nalu
} // namespace Sierra
