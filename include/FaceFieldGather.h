#ifndef FACEFIELDGATHER_H
#define FACEFIELDGATHER_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{

template <int p> face_ordinal_view_t<p> face_entity_offset_to_gid_map(
  const stk::mesh::BulkData&,
  const stk::mesh::Selector&,
  Kokkos::View<int*>);

template <int p> face_view::scalar_view<p> gather_face_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
template <int p> face_view::vector_view<p> gather_face_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);

}}

#endif

