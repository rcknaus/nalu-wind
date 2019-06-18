#ifndef VOLUMES_H
#define VOLUMES_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{

template <int p> ko::scalar_view<p> volumes(ko::vector_view<p> coordinates);
template <int p> ko::scalar_view<p> volumes(ko::scalar_view<p>, ko::vector_view<p> coordinates);



} }

#endif

