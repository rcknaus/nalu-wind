/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ConductionInteriorOperator_h
#define ConductionInteriorOperator_h

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/ConductionKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

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

template <int p> struct ConductionInteriorOperator {
  static constexpr int ndof = 1;
  using op = conduction_element_residual<p>;

  ConductionInteriorOperator(const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    const VectorFieldType& coordField,
    const ScalarFieldType& alpha,
    const ScalarFieldType& diffusivity,
    Kokkos::View<int*> nodeOffsetMap)
  {
    entityOffsets_ = element_entity_offset_to_gid_map<p>(bulk, selector, nodeOffsetMap);
    compute_scaled_geometric_factors(bulk, selector, coordField, alpha, diffusivity);
    set_fields(bulk, selector);
  }

  void compute_scaled_geometric_factors(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& selector,
    const VectorFieldType& coordField,
    const ScalarFieldType& alpha,
    const ScalarFieldType& diffusivity)
  {
    auto coordview = gather_field<p>(bulk, selector, coordField);
    volume_ = volumes<p>(gather_field<p>(bulk, selector, alpha), coordview);
    mapped_area_ = mapped_areas<p>(gather_field<p>(bulk, selector, diffusivity), coordview);
  }

  void set_fields(const stk::mesh::BulkData& bulk,  const stk::mesh::Selector& selector)
  {
    const auto& temperatureField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
    const auto bdf2 = solution_field(bulk).number_of_states() == 3;
    qm1_ = gather_field<p>(bulk, selector, solution_field(bulk).field_of_state((bdf2) ? stk::mesh::StateNM1 : stk::mesh::StateN));
    qp0_ = gather_field<p>(bulk, selector, temperatureField.field_of_state(stk::mesh::StateN));
    qp1_ = predict_state(bulk, selector);
  }

  void set_gamma(Kokkos::Array<double, 3> gammas) { projTimeScale_ = gammas; }

  template <typename TpetraViewType> void compute_linearized_residual(const TpetraViewType& xv, TpetraViewType& yv) const
  {
    for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
      auto delta = gather_delta<p>(index, entityOffsets_, xv);
      const auto delta_view = nodal_scalar_view<p, DoubleType>(delta.data());
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        entityOffsets_,
        op::linearized_residual(index, projTimeScale_[0], volume_, mapped_area_, delta_view),
        yv
      );
    }
  }

  template <typename TpetraViewType> void compute_rhs(TpetraViewType& yv) const
  {
    for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        entityOffsets_,
        op::residual(index, projTimeScale_, volume_, mapped_area_, qm1_, qp0_, qp1_),
        yv
      );
    }
  }

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
        op::residual(index, projTimeScale_, volume_, mapped_area_, qm1_, qp0_, qp1_),
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
    for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
      auto delta = gather_delta<p>(index, entityOffsets_, xv);
      const auto delta_view = nodal_scalar_view<p, DoubleType>(delta.data());
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        maxOwnedRowLid,
        maxSharedNotOwnedLid,
        entityOffsets_,
        op::linearized_residual(index, projTimeScale_[0], volume_, mapped_area_, delta_view),
        yowned,
        yshared
      );
    }
  }

  template <typename TpetraViewType> void compute_matrix_diagonal(TpetraViewType& yv) const
  {
    for (int index = 0; index < entityOffsets_.extent_int(0); ++index) {
      add_element_rhs_to_local_tpetra_vector<p>(
        index,
        entityOffsets_,
        op::lhs_diagonal(index, projTimeScale_[0], volume_, mapped_area_),
        yv
      );
    }
  }

  void cycle_element_views(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
  {
    qm1_ = qp0_;
    qp0_ = qp1_;
    qp1_ = predict_state(bulk, selector);
  }

//  void cycle_element_views(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
//  {
//    auto bdf2 = solution_field(bulk).number_of_states() == 3;
//    qm1_ = gather_field<p>(bulk, selector, solution_field(bulk).field_of_state((bdf2) ? stk::mesh::StateNM1 : stk::mesh::StateN));
//    qp0_ = gather_field<p>(bulk, selector, solution_field(bulk).field_of_state(stk::mesh::StateN));
//    qp1_ = predict_state(bulk, selector);
//  }

  ko::scalar_view<p> predict_state(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
      { return gather_field<p>(bulk, selector,solution_field(bulk).field_of_state(stk::mesh::StateNP1)); }

  const ScalarFieldType& solution_field(const stk::mesh::BulkData& bulk) {
    return *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "temperature");
  }

  Kokkos::Array<double, 3> projTimeScale_{{1.0, -1.0, 0}};
  elem_ordinal_view_t<p> entityOffsets_;
  ko::scalar_view<p> volume_;
  ko::scs_vector_view<p> mapped_area_;
  ko::scalar_view<p> qm1_;
  ko::scalar_view<p> qp0_;
  ko::scalar_view<p> qp1_;

};

} // namespace nalu
} // namespace Sierra

#endif

