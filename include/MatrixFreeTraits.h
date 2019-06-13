/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MFTraits_h
#define MFTraits_h

#include <stk_topology/topology.hpp>

namespace sierra {
namespace nalu {

struct MF {
  static constexpr bool doMatrixFree = true;
  static constexpr int p = 1;
};

} // namespace nalu
} // namespace Sierra

#endif
