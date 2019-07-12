/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NoBoundaryOperator_h
#define NoBoundaryOperator_h

#include "MatrixFreeTypes.h"


namespace sierra { namespace nalu {

class NoOperator {
public:
  template <typename TpetraViewType> void apply(const TpetraViewType&, TpetraViewType&) const {}
  template <typename TpetraViewType> void compute_initial_residual(const TpetraViewType&, TpetraViewType&) const {}

  template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
    global_ordinal_type,
    global_ordinal_type ,
    OwnedViewType,
    SharedViewType) const {}

  template <typename OwnedViewType, typename SharedViewType> void compute_linearized_residual(
    global_ordinal_type,
    global_ordinal_type,
    OwnedViewType,
    OwnedViewType,
    SharedViewType) const {}
};


} // namespace nalu
} // namespace Sierra

#endif

