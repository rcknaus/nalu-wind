#ifndef CVFEMEXPOSEDAREAS_H
#define CVFEMEXPOSEDAREAS_H

#include "CVFEMTypeDefs.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

#include "stk_util/util/ReportHandler.hpp"

namespace sierra { namespace nalu{

template <int p> face_view::vector_view<p> exposed_area_vectors(face_view::vector_view<p>);

}}

#endif

