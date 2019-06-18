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
    const ko::scs_scalar_view<p>& mdot,
    const ko::scalar_view<p>& volume,
    const ko::vector_view<p>& coords,
    const ko::scalar_view<p>& visc,
    const ko::scalar_view<p>& rhom1,
    const ko::scalar_view<p>& rhop0,
    const ko::scalar_view<p>& rhop1,
    const ko::vector_view<p>& velm1,
    const ko::vector_view<p>& velp0,
    const ko::vector_view<p>& velp1,
    const ko::vector_view<p>& Gp);

  static nodal_vector_array<DoubleType, p> linearized_residual(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const ko::scs_scalar_view<p>& mdot,
    const nodal_vector_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> linearized_residual(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const ko::scs_scalar_view<p>& mdot,
    const nodal_scalar_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> lhs_diagonal(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const ko::scs_scalar_view<p>& mdot);

  static constexpr int order = p;
  static constexpr auto solution_field_name = "velocity";
};

} // namespace nalu
} // namespace Sierra

#endif

