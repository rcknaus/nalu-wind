#ifndef ConductionKernel_h
#define ConductionKernel_h

#include "LocalArray.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMTypeDefs.h"
#include "master_element/TensorProductCVFEMOperators.h"

namespace sierra { namespace nalu {

template <int p> struct conduction_element_residual {
  using ftype = DoubleType;

  static nodal_scalar_array<ftype, p> residual(
    int index,
    Kokkos::Array<double, 3> gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const ko::scalar_view<p>& qm1,
    const ko::scalar_view<p>& qp0,
    const ko::scalar_view<p>& qp1);

  static nodal_scalar_array<DoubleType, p> linearized_residual(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const nodal_scalar_view<p, DoubleType>& delta);

  static nodal_scalar_array<ftype, p> lhs_diagonal(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area);

  static constexpr int order = p;
  static constexpr auto solution_field_name = "temperature";
};

} // namespace nalu
} // namespace Sierra

#endif

