/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef PNGBoundaryOperator_h
#define PNGBoundaryOperator_h

#include "MatrixFreeTypes.h"
#include "FaceFieldGather.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMExposedAreas.h"
#include "kernel/ProjectedNodalGradientKernel.h"

namespace sierra { namespace nalu {

template <int p>
class PNGBoundaryOperator {
public:
  using op = png_element_residual<p>;
  PNGBoundaryOperator(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    VectorFieldType& coordField,
    Kokkos::View<int*> nodeOffsetMap)
  {
    faceNodeOffsets_ = face_entity_offset_to_gid_map<p>(bulk, selector, nodeOffsetMap);
    if (faceNodeOffsets_.extent_int(0) > 0) {
      exposedArea_ = exposed_area_vectors<p>(gather_face_field<p>(bulk, selector, coordField));
    }
  }

  void initialize(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    const ScalarFieldType& qBCField)
  {
    if (faceNodeOffsets_.extent_int(0) == 0) { return; }
    qBC_ = gather_face_field<p>(bulk, selector, qBCField);
  }

  template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    if (faceNodeOffsets_.extent_int(0) == 0) { return; }

    for (int index = 0; index < faceNodeOffsets_.extent_int(0); ++index) {
      add_face_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        faceNodeOffsets_,
        op::boundary_closure(index, exposedArea_, qBC_),
        yowned,
        yshared
      );
    }
  }

  template <typename VectorViewType> void compute_linearized_residual(
    global_ordinal_type,
    global_ordinal_type,
    const VectorViewType&,
    VectorViewType&,
    VectorViewType&) const
  {}

private:
  face_ordinal_view_t<p> faceNodeOffsets_;
  face_view::vector_view<p> exposedArea_;
  face_view::scalar_view<p> qBC_;
};


} // namespace nalu
} // namespace Sierra

#endif

