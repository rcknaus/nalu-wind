/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumInteriorOperator_h
#define MomentumInteriorOperator_h

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "CVFEMCorrectedMassFlux.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"
#include "MatrixFreeTypes.h"
#include "kernel/MomentumKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

namespace sierra { namespace nalu {

template <int p> struct MomentumInteriorOperator {
  static constexpr int ndof = 3;
  using op = momentum_element_residual<p>;

  MomentumInteriorOperator(
    const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    const VectorFieldType& coordField,
    Kokkos::View<int*> nodeOffsetMap,
    bool variableProp = true) : variableProp_(variableProp)
  {
    entityOffsets_ = element_entity_offset_to_gid_map<p>(bulk, selector, nodeOffsetMap);
    coords_ = gather_field<p>(bulk, selector, coordField);
    volume_ = volumes<p>(coords_);
  }

  MomentumInteriorOperator() = default;

  void initialize(const stk::mesh::BulkData& bulk, stk::mesh::Selector selector)
  {
    const auto bdf2 = solution_field(bulk).number_of_states() == 3;
    velm1_ = gather_field<p>(bulk, selector,
      solution_field(bulk).field_of_state((bdf2) ? stk::mesh::StateNM1 : stk::mesh::StateN));
    velp0_ = gather_field<p>(bulk, selector, solution_field(bulk).field_of_state(stk::mesh::StateN));
    velp1_ = predict_state(bulk, selector);

    auto& meta = bulk.mesh_meta_data();
    const auto& rhoField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
    rhom1_ = gather_field<p>(bulk, selector,
      rhoField.field_of_state((bdf2) ? stk::mesh::StateNM1 : stk::mesh::StateN));
    rhop0_ = gather_field<p>(bulk, selector, rhoField.field_of_state(stk::mesh::StateN));
    rhop1_ = gather_field<p>(bulk, selector, rhoField.field_of_state(stk::mesh::StateNP1));

    const auto& GpField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
    Gp_ = gather_field<p>(bulk, selector, GpField.field_of_state(stk::mesh::StateNone));

    const auto& viscField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
    visc_ = gather_field<p>(bulk, selector, viscField.field_of_state(stk::mesh::StateNone));

    const auto& pressField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
    pressure_ = gather_field<p>(bulk, selector, pressField.field_of_state(stk::mesh::StateNone));

    scaled_volume_ = volumes<p>(rhop1_, coords_);
    mapped_area_ = mapped_areas<p>(visc_, coords_);
  }

  void update_element_views(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
  {
    initialize(bulk, selector);
//    const auto bdf2 = solution_field(bulk).number_of_states() == 3;
//    velm1_ = gather_field<p>(bulk, selector,
//      solution_field(bulk).field_of_state((bdf2) ? stk::mesh::StateNM1 : stk::mesh::StateN));
//    velp0_ = gather_field<p>(bulk, selector, solution_field(bulk).field_of_state(stk::mesh::StateN));
//    velp1_ = predict_state(bulk, selector);
//
//    auto& meta = bulk.mesh_meta_data();
//    const auto& GpField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
//    Gp_ = gather_field<p>(bulk, selector, GpField.field_of_state(stk::mesh::StateNone));
//
//    const auto& pressField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
//    pressure_ = gather_field<p>(bulk, selector, pressField.field_of_state(stk::mesh::StateNone));
  }

//  void cycle_element_views(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
//  {
//    velm1_ = velp0_;
//    velp0_ = velp1_;
//    velp1_ = predict_state(bulk, selector);
//
//    auto& meta = bulk.mesh_meta_data();
//    const auto& GpField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
//    Gp_ = gather_field<p>(bulk, selector, GpField.field_of_state(stk::mesh::StateNone));
//
//    if (variableProp_) {
//      rhom1_ = rhop0_;
//      rhop0_ = rhop1_;
//      auto& rhoField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
//      rhop1_ = gather_field<p>(bulk, selector, rhoField.field_of_state(stk::mesh::StateNP1));
//
//      // fixme: needs to change for LES
//      const auto& viscField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
//      visc_ = gather_field<p>(bulk, selector, viscField.field_of_state(stk::mesh::StateNone));
//
//      scaled_volume_ = volumes<p>(rhop1_, coords_);
//      mapped_area_ = mapped_areas<p>(visc_, coords_);
//    }
//  }

  void compute_mdot(double projTimeScale)
  {
    mdot_ = corrected_mass_flux<p>(projTimeScale, coords_, rhop1_, velp1_, pressure_, Gp_);
  }

  void set_gamma(Kokkos::Array<double, 3> gammas) { gamma_ = gammas; }

  template <typename OwnedViewType, typename SharedViewType> void compute_rhs(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        entityOffsets_,
        op::residual(index, gamma_, mdot_, volume_, coords_, visc_,
          rhom1_, rhop0_, rhop1_, velm1_, velp0_, velp1_, Gp_),
        yowned,
        yshared
      );
    }
  }

  template <typename OwnedViewType, typename SharedViewType> void compute_linearized_residual(
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedLid,
    OwnedViewType xv,
    OwnedViewType yowned,
    SharedViewType yshared) const
  {
    if (xv.extent_int(1) == 3) {
      for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
        auto delta = gather_vector_delta<p>(index, entityOffsets_, xv);
        const auto delta_view = nodal_vector_view<p, DoubleType>(delta.data());
        add_element_rhs_to_local_tpetra_vector<p>(
          index,
          maxOwnedRowLid,
          maxSharedNotOwnedLid,
          entityOffsets_,
          op::linearized_residual(index, gamma_[0], volume_, mapped_area_, mdot_, delta_view),
          yowned,
          yshared
        );
      }
    }
    else{
      for (int d = 0; d < xv.extent_int(1); ++d) {
        for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
          auto delta = gather_delta<p>(index, entityOffsets_, xv, d);
          const auto delta_view = nodal_scalar_view<p, DoubleType>(delta.data());
          add_element_rhs_to_local_tpetra_vector<p>(
            index,
            maxOwnedRowLid,
            maxSharedNotOwnedLid,
            entityOffsets_,
            op::linearized_residual(index, gamma_[0], volume_, mapped_area_, mdot_, delta_view),
            yowned,
            yshared,
            d
          );
        }
      }
    }
  }

  void set_mdot(ko::scs_scalar_view<p> mdot) { mdot_ =  mdot; }

  ko::vector_view<p> predict_state(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
      { return gather_field<p>(bulk, selector,solution_field(bulk).field_of_state(stk::mesh::StateNP1)); }

  const VectorFieldType& solution_field(const stk::mesh::BulkData& bulk) {
    return *bulk.mesh_meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  }

  const bool variableProp_{true};
  Kokkos::Array<double, 3> gamma_{{1.0, -1.0, 0}};
  elem_ordinal_view_t<p> entityOffsets_;
  ko::vector_view<p> coords_;
  ko::scalar_view<p> volume_;
  ko::scalar_view<p> scaled_volume_;
  ko::scs_vector_view<p> mapped_area_;
  ko::scs_scalar_view<p> mdot_;

  //temp
  ko::scalar_view<p> pressure_;

  ko::scalar_view<p> visc_;
  ko::scalar_view<p> rhom1_;
  ko::scalar_view<p> rhop0_;
  ko::scalar_view<p> rhop1_;
  ko::vector_view<p> Gp_;
  ko::vector_view<p> velm1_;
  ko::vector_view<p> velp0_;
  ko::vector_view<p> velp1_;

};

} // namespace nalu
} // namespace Sierra

#endif

