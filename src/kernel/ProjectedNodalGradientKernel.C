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
  const elem_view::scalar_view<p>& volume,
  const elem_view::scs_vector_view<p>& area,
  const elem_view::scalar_view<p>& q,
  const elem_view::vector_view<p>& dqdx)
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
  const elem_view::scalar_view<p>& volume,
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
  const elem_view::scalar_view<p>& volume,
  const nodal_scalar_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);

  auto elem_vol = nodal_scalar_view<p, ftype>(&volume(index, 0, 0, 0));
  tensor_assembly::green_gauss_rhs_linearized(ops, elem_vol, delta, v_rhs);
  return rhs;
}
namespace {
template <int p, typename ftype> void face_area_integral(
  const CVFEMOperators<p, ftype>& ops,
  const face_nodal_vector_array<ftype,p>& integrand,
  face_nodal_vector_array<ftype, p>& rhs)
{
  static constexpr int n1D = p + 1;
  const auto& W = ops.mat_.nodalWeights;
  auto scratch = face_nodal_vector_array<ftype, p>();
  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      ftype acc_x = 0;
      ftype acc_y = 0;
      ftype acc_z = 0;
      for (int q = 0; q < n1D; ++q) {
        const auto temp = W(i, q);
        acc_x += temp * integrand(j, q, XH);
        acc_y += temp * integrand(j, q, YH);
        acc_z += temp * integrand(j, q, ZH);
      }
      scratch(j, i, XH) = acc_x;
      scratch(j, i, YH) = acc_y;
      scratch(j, i, ZH) = acc_z;
    }
  }

  for (int j = 0; j < n1D; ++j) {

    for (int i = 0; i < n1D; ++i) {
      rhs(j, i, XH) = 0;
      rhs(j, i, YH) = 0;
      rhs(j, i, ZH) = 0;
    }

    for (int q = 0; q < n1D; ++q) {
      const auto temp = W(j, q);
      for (int i = 0; i < n1D; ++i) {
        rhs(j, i, XH) += temp * scratch(q, i, XH);
        rhs(j, i, YH) += temp * scratch(q, i, YH);
        rhs(j, i, ZH) += temp * scratch(q, i, ZH);
      }
    }
  }
}
}

template <int p> face_nodal_vector_array<DoubleType, p> png_element_residual<p>::boundary_closure(
  int index,
  const face_view::vector_view<p>& area,
  const face_view::scalar_view<p>& qBC)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = face_nodal_vector_array<ftype, p>();
  auto integrand = face_nodal_vector_array<ftype,p>();

  static constexpr int n1D = p + 1;
  for (int j = 0; j < n1D; ++j) {
    for (int i = 0; i < n1D; ++i) {
      const auto qVal = qBC(index, j, i);
      integrand(j, i, XH) = qVal * area(index, j, i, XH);
      integrand(j, i, YH) = qVal * area(index, j, i, YH);
      integrand(j, i, ZH) = qVal * area(index, j, i, ZH);
    }
  }
  face_area_integral(ops, integrand, rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> png_element_residual<p>::lhs_diagonal(
  int index, const elem_view::scalar_view<p>& volume)
{
  static const auto ops = CVFEMOperators<p>();

  constexpr bool lumped = false;
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

