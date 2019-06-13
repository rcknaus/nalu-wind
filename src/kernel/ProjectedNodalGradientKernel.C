/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include "kernel/ProjectedNodalGradientKernel.h"

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include <kernel/TensorProductCVFEMPNG.h>

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

  auto elem_area = scs_vector_view<p, ftype>(&area(index,0,0,0,0,0));
  auto elem_vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));

  auto elem_q = nodal_scalar_view<p, ftype>(&q(index,0,0,0));
  auto elem_dqdx = nodal_vector_view<p, ftype>(&dqdx(index,0,0,0,0));

  tensor_assembly::green_gauss_rhs(ops, elem_area, elem_vol, elem_q, elem_dqdx, v_rhs);
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

  auto lhs = la::zero<matrix_array<ftype, p>>();
  auto v_lhs = la::make_view(lhs);

  constexpr int n1D = p + 1;
  auto diag = nodal_scalar_array<ftype, p>();
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        diag(k, j, i) = volume(index, k, j, i);
      }
    }
  }
  return diag;
}
template class png_element_residual<1>;
template class png_element_residual<2>;
template class png_element_residual<3>;
template class png_element_residual<4>;

} // namespace nalu
} // namespace Sierra

