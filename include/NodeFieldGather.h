#ifndef NODEFIELDGATHER_H
#define NODEFIELDGATHER_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{

Kokkos::View<int*> node_offset_view(const stk::mesh::BulkData&, const stk::mesh::Selector&, Kokkos::View<int*>);
node_view::scalar_view<> gather_node_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
node_view::vector_view<> gather_node_field(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);

}}

#endif

