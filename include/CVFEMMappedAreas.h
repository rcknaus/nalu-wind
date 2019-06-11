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

template <int p> ko::scs_vector_view<p> mapped_areas(ko::vector_view<p>);
template <int p> ko::scs_vector_view<p> mapped_areas(ko::scalar_view<p>, ko::vector_view<p>);
}}

#endif

