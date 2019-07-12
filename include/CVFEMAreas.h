#ifndef CVFEMAREAS_H
#define CVFEMAREAS_H

#include "CVFEMTypeDefs.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

#include "master_element/TensorProductCVFEMVolumeMetric.h"
#include "element_promotion/NodeMapMaker.h"

#include "stk_util/util/ReportHandler.hpp"

namespace sierra { namespace nalu{

template <int p> elem_view::scs_vector_view<p> area_vectors(elem_view::vector_view<p>);

}}

#endif

