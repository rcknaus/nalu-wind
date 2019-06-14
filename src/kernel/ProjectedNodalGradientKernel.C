/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include "kernel/ProjectedNodalGradientKernel.h"

#include "MatrixFreeTraits.h"

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include "kernel/TensorProductCVFEMPNG.h"

namespace sierra { namespace nalu {

template <int p> nodal_vector_array<DoubleType, p> png_element_residual<p>::residual(
  int index,
  const ko::scalar_view<p>& volume,
  const ko::scs_vector_view<p>& area,
  const ko::scalar_view<p>& q,
  const ko::vector_view<p>& dqdx)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_vector_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);

  const auto elem_area = scs_vector_view<p>(&area(index, 0, 0, 0, 0, 0));
  const auto elem_volume = nodal_scalar_view<p>(&volume(index, 0, 0, 0));
  const auto elem_q = nodal_scalar_view<p>(&q(index, 0, 0, 0));
  const auto elem_dqdx = nodal_vector_view<p>(&dqdx(index, 0, 0, 0, 0));
  tensor_assembly::green_gauss_rhs(ops, elem_area, elem_volume, elem_q, elem_dqdx, v_rhs);
  return rhs;
}

template <int p> nodal_vector_array<DoubleType, p>
png_element_residual<p>::linearized_residual(
  int index,
  const ko::scalar_view<p>& volume,
  const nodal_vector_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_vector_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);

  auto elem_vol = nodal_scalar_view<p, ftype>(&volume(index, 0, 0, 0));
  tensor_assembly::green_gauss_rhs_linearized(ops, elem_vol, delta, v_rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> png_element_residual<p>::linearized_residual(
  int index,
  const ko::scalar_view<p>& volume,
  const nodal_scalar_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);

  auto elem_vol = nodal_scalar_view<p, ftype>(&volume(index, 0, 0, 0));
  tensor_assembly::green_gauss_rhs_linearized(ops, elem_vol, delta, v_rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> png_element_residual<p>::lhs_diagonal(
  int index, const ko::scalar_view<p>& volume)
{
  static const auto ops = CVFEMOperators<p>();

  bool lumped = true;
  const auto& weights = (lumped) ? ops.mat_.nodalWeights : ops.mat_.lumpedNodalWeights;

  constexpr int n1D = p + 1;
  auto diag = nodal_scalar_array<ftype, p>();
  for (int k = 0; k < n1D; ++k) {
    const auto Wk = weights(k,k);
    for (int j = 0; j < n1D; ++j) {
      const auto WjWk = Wk * weights(j,j);
      for (int i = 0; i < n1D; ++i) {
        diag(k, j, i) = WjWk * weights(i, i) * volume(index, k, j, i);
      }
    }
  }
  return diag;
}
template class png_element_residual<POLY1>;
template class png_element_residual<POLY2>;
template class png_element_residual<POLY3>;
template class png_element_residual<POLY4>;

} // namespace nalu
} // namespace Sierra

