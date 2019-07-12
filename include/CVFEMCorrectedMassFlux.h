#ifndef CVFEMMCorrectMassFlux_H
#define CVFEMMCorrectMassFlux_H

#include "CVFEMTypeDefs.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

namespace sierra { namespace nalu{

template <int p> elem_view::scs_scalar_view<p> corrected_mass_flux(
  double projTimeScale,
  elem_view::vector_view<p> coords,
  elem_view::scalar_view<p> rho,
  elem_view::vector_view<p> velocity,
  elem_view::scalar_view<p> pressure,
  elem_view::vector_view<p> projGradP);

}}

#endif

