#ifndef ContinuityKernel_h
#define ContinuityKernel_h

#include "LocalArray.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMTypeDefs.h"
#include "master_element/TensorProductCVFEMOperators.h"

namespace sierra { namespace nalu {

template <int p> struct continuity_element_residual {
  using ftype = DoubleType;

  static nodal_scalar_array<ftype, p> rhs(
    int index,
    double projTimeScale,
    const elem_view::scs_scalar_view<p>& mdot,
    const elem_view::scalar_view<p>& qp1);

  static nodal_scalar_array<DoubleType, p> linearized_residual(
    int index,
    const elem_view::scs_vector_view<p>& mapped_area,
    const nodal_scalar_view<p, DoubleType>& delta);

  static constexpr int order = p;
  static constexpr auto solution_field_name = "pressure";
};

} // namespace nalu
} // namespace Sierra

#endif

