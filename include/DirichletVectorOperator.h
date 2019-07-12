/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef DirichletVectorOperator_h
#define DirichletVectorOperator_h

#include "MatrixFreeTypes.h"
#include "NodeFieldGather.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"


namespace sierra { namespace nalu {

class DirichletVectorOperator {
public:
  DirichletVectorOperator(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    Kokkos::View<int*> nodeOffsetMap)
  {
    dirichletNodeOffsets_ = node_offset_view(bulk, selector, nodeOffsetMap);
  }

  void initialize(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    const VectorFieldType& solnField,
    const VectorFieldType& bcField)
  {
    if (dirichletNodeOffsets_.extent_int(0) == 0) { return; }

    uSoln_ = gather_node_field(bulk, selector,solnField);
    uBC_ = gather_node_field(bulk, selector, bcField);
  }

  template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    if (dirichletNodeOffsets_.extent_int(0) == 0) { return; }

    for (int index  = 0; index < dirichletNodeOffsets_.extent_int(0); ++index) {
      const auto rowLid = dirichletNodeOffsets_(index);
      if (rowLid < maxOwnedRowLid) {
        yowned(rowLid, 0) = -(uBC_(index, 0) - uSoln_(index, 0));
        yowned(rowLid, 1) = -(uBC_(index, 1) - uSoln_(index, 1));
        yowned(rowLid, 2) = -(uBC_(index, 2) - uSoln_(index, 2));
      }
      else if (rowLid < maxSharedNotOwnedLid) {
        const auto actualID = rowLid - maxOwnedRowLid;
        yshared(actualID, 0) = 0;
        yshared(actualID, 1) = 0;
        yshared(actualID, 2) = 0;
      }
    }
  }

  template <typename VectorViewType> void compute_linearized_residual(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    const VectorViewType& xv,
    VectorViewType& yowned,
    VectorViewType& yshared) const
  {
    if (dirichletNodeOffsets_.extent_int(0) == 0) { return; }

    const auto numLHS = xv.extent_int(1);
    ThrowRequire(numLHS == yowned.extent_int(1) && numLHS == yshared.extent_int(1));
    for (int index  = 0; index < dirichletNodeOffsets_.extent_int(0); ++index) {
      const auto rowLid = dirichletNodeOffsets_(index);
      if (rowLid < maxOwnedRowLid) {
        for (int d = 0; d < numLHS; ++d) {
          yowned(rowLid, d) = -xv(rowLid, d);
        }
      }
      else if (rowLid < maxSharedNotOwnedLid) {
        const auto actualID = rowLid - maxOwnedRowLid;
        for (int d = 0; d < numLHS; ++d) {
          yshared(actualID, d) = 0;
        }
      }
    }
  }

private:
  Kokkos::View<int*> dirichletNodeOffsets_;
  node_view::vector_view<> uSoln_;
  node_view::vector_view<> uBC_;
};


} // namespace nalu
} // namespace Sierra

#endif

