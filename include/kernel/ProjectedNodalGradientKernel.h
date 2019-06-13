#ifndef ProjectedNodalGradientKernel_h
#define ProjectedNodalGradientKernel_h

#include "LocalArray.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMTypeDefs.h"
#include "master_element/TensorProductCVFEMOperators.h"

namespace sierra { namespace nalu {

template <int p> struct png_element_residual {
  using ftype = DoubleType;

  static nodal_vector_array<ftype, p> residual(
    int index,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& area,
    const ko::scalar_view<p>& q,
    const ko::vector_view<p>& dqdx);

  static nodal_vector_array<DoubleType, p> linearized_residual(
    int index,
    const ko::scalar_view<p>& volume,
    const nodal_vector_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> linearized_residual(
    int index,
    const ko::scalar_view<p>& volume,
    const nodal_scalar_view<p, DoubleType>& delta);

  static nodal_scalar_array<DoubleType, p> lhs_diagonal(
    int index,
    const ko::scalar_view<p>& volume);

  static constexpr int order = p;
  static constexpr auto solution_field_name = "png";
};

} // namespace nalu
} // namespace Sierra

#endif

