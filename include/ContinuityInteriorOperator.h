/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ContinuityInteriorOperator_h
#define ContinuityInteriorOperator_h

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"
#include "CVFEMCorrectedMassFlux.h"
#include "MatrixFreeTypes.h"
#include "kernel/ContinuityKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

namespace sierra { namespace nalu {

template <int p> struct ContinuityInteriorOperator {
  static constexpr int ndof = 1;
  using op = continuity_element_residual<p>;

  ContinuityInteriorOperator(const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    const VectorFieldType& coordField,
    Kokkos::View<int*> nodeOffsetMap)
  {
    entityOffsets_ = element_entity_offset_to_gid_map<p>(bulk, selector, nodeOffsetMap);
    coords_ = gather_field<p>(bulk, selector, coordField);
    mapped_area_ = mapped_areas<p>(coords_);
    initialize(bulk, selector);
  }

  void initialize(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
  {
    const auto& meta = bulk.mesh_meta_data();
    const auto& rhoField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
    rhop1_ = gather_field<p>(bulk, selector, rhoField.field_of_state(stk::mesh::StateNP1));

    const auto& velField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
    velp1_ = gather_field<p>(bulk, selector, velField.field_of_state(stk::mesh::StateNP1));

    const auto& GpField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
    Gp_ = gather_field<p>(bulk, selector, GpField.field_of_state(stk::mesh::StateNone));

    const auto& pressField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
    pressure_ = gather_field<p>(bulk, selector, pressField.field_of_state(stk::mesh::StateNone));
  }

  void compute_mdot(double projTimeScale)
  {
    mdot_ = corrected_mass_flux<p>(projTimeScale, coords_, rhop1_, velp1_, pressure_, Gp_);
  }

  void set_projected_timescale(double projTimeScale) { projTimeScale_ = projTimeScale; }

  using HostRangePolicy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, int>;

  template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    const auto range = HostRangePolicy(0, entityOffsets_.extent_int(0));
    Kokkos::parallel_for("continuity_rhs", range, KOKKOS_LAMBDA(const int index) {
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        entityOffsets_,
        op::rhs(index, projTimeScale_, mdot_, pressure_),
        yowned,
        yshared
      );
    });
  }

  template <typename OwnedViewType, typename SharedViewType> void compute_linearized_residual(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType xv,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    const auto range = HostRangePolicy(0, entityOffsets_.extent_int(0));
    Kokkos::parallel_for("continuity_residual", range, KOKKOS_LAMBDA(const int index) {
      auto delta = gather_delta<p>(index, entityOffsets_, xv);
      const auto delta_view = nodal_scalar_view<p, DoubleType>(delta.data());
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        entityOffsets_,
        op::linearized_residual(index, mapped_area_, delta_view),
        yowned,
        yshared
      );
    });
  }

  void cycle_element_views(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
  {
    pressure_ = predict_state(bulk, selector);
  }

  elem_view::scalar_view<p> predict_state(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
      { return gather_field<p>(bulk, selector,solution_field(bulk).field_of_state(stk::mesh::StateNone)); }

  const ScalarFieldType& solution_field(const stk::mesh::BulkData& bulk) {
    return *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  }

  double projTimeScale_{1.0};
  elem_ordinal_view_t<p> entityOffsets_;
  elem_view::vector_view<p> coords_;
  elem_view::scs_scalar_view<p> mdot_;
  elem_view::scs_vector_view<p> mapped_area_;
  elem_view::scalar_view<p> rhop1_;
  elem_view::vector_view<p> velp1_;
  elem_view::vector_view<p> Gp_;
  elem_view::scalar_view<p> pressure_;
};

} // namespace nalu
} // namespace Sierra

#endif

