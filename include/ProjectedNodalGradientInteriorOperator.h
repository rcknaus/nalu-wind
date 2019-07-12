/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ProjectedNodalGradientInteriorOperator_h
#define ProjectedNodalGradientInteriorOperator_h

#include "SimdFieldGather.h"
#include "CVFEMAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/ProjectedNodalGradientKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

namespace sierra { namespace nalu {

template <int p> struct ProjectedNodalGradientInteriorOperator {
  using op = png_element_residual<p>;

ProjectedNodalGradientInteriorOperator(const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  const VectorFieldType& coordField,
  Kokkos::View<int*> nodeOffsetMap)
{
  entityOffsets_ = element_entity_offset_to_gid_map<p>(bulk, selector, nodeOffsetMap);
  auto coordview = gather_field<p>(bulk, selector, coordField);
  volume_ = volumes<p>(coordview);
  area_ = area_vectors<p>(coordview);
}

void initialize(const stk::mesh::BulkData& bulk,  const stk::mesh::Selector& selector,
  const ScalarFieldType& qField,
  const VectorFieldType& dqdxField)
{
  q_ = gather_field<p>(bulk, selector, qField);
  dqdx_ = gather_field<p>(bulk, selector, dqdxField);
}

using HostRangePolicy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, int>;


template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
  global_ordinal_type maxOwnedRowLid,
  global_ordinal_type maxSharedNotOwnedLid,
  OwnedViewType yowned,
  SharedViewType yshared) const
{
  const auto range = HostRangePolicy(0, entityOffsets_.extent_int(0));
  Kokkos::parallel_for("png_vector_rhs", range, KOKKOS_LAMBDA(const int index) {
    add_element_rhs_to_local_tpetra_vector<p>(
      index,
      maxOwnedRowLid,
      maxSharedNotOwnedLid,
      entityOffsets_,
      op::residual(index, volume_, area_, q_, dqdx_),
      yowned,
      yshared
    );
  });
}

template <typename OwnedViewType, typename SharedViewType>
void compute_linearized_residual(
  global_ordinal_type maxOwnedRowLid,
  global_ordinal_type maxSharedNotOwnedLid,
  OwnedViewType xv,
  OwnedViewType yowned,
  SharedViewType yshared) const
{
  const auto range = HostRangePolicy(0, entityOffsets_.extent_int(0));
  if (xv.extent_int(1) == 3) {
    Kokkos::parallel_for("png_vector_residual", range, KOKKOS_LAMBDA(const int index) {
      auto delta = gather_vector_delta<p>(index, entityOffsets_, xv);
      const auto delta_view = nodal_vector_view<p, DoubleType>(delta.data());
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        entityOffsets_,
        op::linearized_residual(index, volume_, delta_view),
        yowned,
        yshared
      );
    });
  }
  else {
    Kokkos::parallel_for("png_residual", range, KOKKOS_LAMBDA(const int index) {
      for (int d = 0; d < xv.extent_int(1); ++d) {
        auto delta = gather_delta<p>(index, entityOffsets_, xv, d);
        const auto delta_view = nodal_scalar_view<p, DoubleType>(delta.data());
        add_element_rhs_to_local_tpetra_vector<p>(
          index,
          maxOwnedRowLid,
          maxSharedNotOwnedLid,
          entityOffsets_,
          op::linearized_residual(index, volume_, delta_view),
          yowned,
          yshared,
          d
        );
      }
    });
  }
}

elem_ordinal_view_t<p> entityOffsets_;
elem_view::scalar_view<p> volume_;
elem_view::scs_vector_view<p> area_;
elem_view::scalar_view<p> q_;
elem_view::vector_view<p> dqdx_;
};

} // namespace nalu
} // namespace Sierra

#endif

