#ifndef CVFEMMEffVisc_H
#define CVFEMMEffVisc_H

#include "CVFEMTypeDefs.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

namespace sierra { namespace nalu{

template <int p> elem_view::scalar_view<p> nodal_smagorinsky_viscosity(
  double smagConstant,
  elem_view::vector_view<p> coords,
  elem_view::scalar_view<p> density,
  elem_view::scalar_view<p> visc,
  elem_view::scalar_view<p> dnv,
  elem_view::vector_view<p> velocity);

template <int p> elem_view::scalar_view<p> nodal_wale_viscosity(
  double waleConstant,
  elem_view::vector_view<p> coords,
  elem_view::scalar_view<p> density,
  elem_view::scalar_view<p> visc,
  elem_view::scalar_view<p> dnv,
  elem_view::vector_view<p> velocity);

}}

#endif

