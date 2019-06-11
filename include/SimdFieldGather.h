#ifndef SIMDFIELDGATHER_H
#define SIMDFIELDGATHER_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{


static constexpr int invalid_node_index = -1;
inline bool valid_index(int k) { return k != invalid_node_index; }

template <int p> elem_ordinal_view_t<p> element_entity_offset_to_gid_map(
  const stk::mesh::BulkData&,
  const stk::mesh::Selector&,
  Kokkos::View<int*>);

template <int p> elem_ordinal_view_t<p> test_element_entity_offset_to_gid_map(
  const stk::mesh::BulkData&, const stk::mesh::Selector&);

template <int p> elem_entity_view_t<p> element_entity_view(
  const stk::mesh::BulkData&, const stk::mesh::Selector&);


template <int p> void write_to_stk_field(
  const stk::mesh::BulkData&,
  elem_ordinal_view_t<p>,
  ko::scalar_view<p>,
  ScalarFieldType&);

template <int p> ko::scalar_view<p> gather_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
template <int p> ko::vector_view<p> gather_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);

} }

#endif

