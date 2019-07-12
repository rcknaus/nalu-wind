#ifndef MomentumKernel_h
#define MomentumKernel_h

#include "LocalArray.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMTypeDefs.h"
#include "master_element/TensorProductCVFEMOperators.h"

namespace sierra { namespace nalu {

template <int p> struct momentum_element_residual {
  using ftype = DoubleType;

  static nodal_vector_array<ftype, p> residual(
    int index,
    Kokkos::Array<double, 3> gamma,
    const elem_view::scs_scalar_view<p>& mdot,
    const elem_view::scalar_view<p>& volume,
    const elem_view::vector_view<p>& coords,
    const elem_view::scalar_view<p>& visc,
    const elem_view::scalar_view<p>& rhom1,
    const elem_view::scalar_view<p>& rhop0,
    const elem_view::scalar_view<p>& rhop1,
    const elem_view::vector_view<p>& velm1,
    const elem_view::vector_view<p>& velp0,
    const elem_view::vector_view<p>& velp1,
    const elem_view::vector_view<p>& Gp);

  static nodal_vector_array<DoubleType, p> linearized_residual(
    int index,
    double gamma,
    const elem_view::scalar_view<p>& volume,
    const elem_view::scs_vector_view<p>& area,
    const elem_view::scs_scalar_view<p>& mdot,
    const nodal_vector_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> linearized_residual(
    int index,
    double gamma,
    const elem_view::scalar_view<p>& volume,
    const elem_view::scs_vector_view<p>& area,
    const elem_view::scs_scalar_view<p>& mdot,
    const nodal_scalar_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> lhs_diagonal(
    int index,
    double gamma,
    const elem_view::scalar_view<p>& volume,
    const elem_view::scs_vector_view<p>& area,
    const elem_view::scs_scalar_view<p>& mdot);

  static constexpr int order = p;
  static constexpr auto solution_field_name = "velocity";
};

} // namespace nalu
} // namespace Sierra

#endif

