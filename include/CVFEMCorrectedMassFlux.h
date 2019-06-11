#ifndef CVFEMMCorrectMassFlux_H
#define CVFEMMCorrectMassFlux_H

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu{

template <int p> ko::scs_scalar_view<p> corrected_mass_flux(
  double projTimeScale,
  ko::vector_view<p> coords,
  ko::scalar_view<p> rho,
  ko::vector_view<p> velocity,
  ko::scalar_view<p> pressure,
  ko::vector_view<p> projGradP);

}}

#endif

