#ifndef CVFEMMAPPEDAREAS_H
#define CVFEMMAPPEDAREAS_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{

template <int p> elem_view::scs_vector_view<p> mapped_areas(elem_view::vector_view<p>);
template <int p> elem_view::scs_vector_view<p> mapped_areas(elem_view::scalar_view<p>, elem_view::vector_view<p>);
}}

#endif

