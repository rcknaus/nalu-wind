/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMPNG_h
#define TensorProductCVFEMPNG_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int poly_order, typename Scalar>
void green_gauss_lhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  matrix_vector_view<poly_order, Scalar>& lhs)
{
  constexpr int n1D = poly_order + 1;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        const int rowIndices[3] = {
            idx<n1D>(XH, n, m, l),
            idx<n1D>(YH, n, m, l),
            idx<n1D>(ZH, n, m, l)
        };

        for (int k = 0; k < n1D; ++k) {
          const auto Wnk = weight(n, k);
          for (int j = 0; j < n1D; ++j) {
            auto WnkWmj = Wnk * weight(m, j);
            for (int i = 0; i < n1D; ++i) {
              const auto lhsfac = WnkWmj * weight(l, i) * vol(k, j, i);
              lhs(rowIndices[XH], idx<n1D>(XH, k, j, i)) += lhsfac;
              lhs(rowIndices[YH], idx<n1D>(YH, k, j, i)) += lhsfac;
              lhs(rowIndices[ZH], idx<n1D>(ZH, k, j, i)) += lhsfac;
            }
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void green_gauss_lhs_lumped(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  matrix_vector_view<poly_order, Scalar>& lhs)
{
  constexpr int n1D = poly_order + 1;
  const auto& weight = ops.mat_.lumpedNodalWeights;

  for (int n = 0; n < n1D; ++n) {
    const auto Wn = weight(n,n);
    for (int m = 0; m < n1D; ++m) {
      const auto WnWm = Wn * weight(m,m);
      for (int l = 0; l < n1D; ++l) {
        const auto lhsfac = WnWm * weight(l,l) * vol(n,m,l);
        const int rowIndices[3] = {
            idx<n1D>(XH, n, m, l),
            idx<n1D>(YH, n, m, l),
            idx<n1D>(ZH, n, m, l)
        };

        lhs(rowIndices[XH], rowIndices[XH]) += lhsfac;
        lhs(rowIndices[YH], rowIndices[YH]) += lhsfac;
        lhs(rowIndices[ZH], rowIndices[ZH]) += lhsfac;
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int poly_order, typename Scalar>
void green_gauss_rhs(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const scs_vector_view<poly_order, Scalar>& area,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  const nodal_scalar_view<poly_order, Scalar>& q,
  const nodal_vector_view<poly_order, Scalar>& dqdx,
  nodal_vector_view<poly_order, Scalar>& rhs)
{
  constexpr int n1D = poly_order + 1;
  constexpr int nscs = poly_order;

  nodal_vector_array<Scalar, poly_order> work_integrand;
  auto integrand = la::make_view(work_integrand);

  nodal_scalar_array<Scalar, poly_order> work_q_scs;
  auto q_scs = la::make_view(work_q_scs);

  ops.scs_xhat_interp(q, q_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < nscs; ++i) {
        const auto q_val = q_scs(k, j, i);
//        std::cout << "areax: " <<  area(XH, k, j, i, XH) << ", "<< area(XH, k, j, i, YH) << ", " << area(XH, k, j, i, ZH) << std::endl;
        integrand(k, j, i, XH) = area(XH, k, j, i, XH) * q_val;
        integrand(k, j, i, YH) = area(XH, k, j, i, YH) * q_val;
        integrand(k, j, i, ZH) = area(XH, k, j, i, ZH) * q_val;
      }
    }
  }
  ops.integrate_and_diff_xhat(integrand, rhs);

  ops.scs_yhat_interp(q, q_scs);
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < nscs; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto q_val = q_scs(k, j, i);
//        std::cout << "areay: " <<  area(YH, k, j, i, XH) << ", "<< area(YH, k, j, i, YH) << ", " << area(YH, k, j, i, ZH) << std::endl;

        integrand(k, j, i, XH) = area(YH, k, j, i, XH) * q_val;
        integrand(k, j, i, YH) = area(YH, k, j, i, YH) * q_val;
        integrand(k, j, i, ZH) = area(YH, k, j, i, ZH) * q_val;
      }
    }
  }
  ops.integrate_and_diff_yhat(integrand, rhs);

  ops.scs_zhat_interp(q, q_scs);
  for (int k = 0; k < nscs; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto q_val = q_scs(k, j, i);
//        std::cout << "areaz: " <<  area(ZH, k, j, i, XH) << ", "<< area(ZH, k, j, i, YH) << ", " << area(ZH, k, j, i, ZH) << std::endl;

        integrand(k, j, i, XH) = area(ZH, k, j, i, XH) * q_val;
        integrand(k, j, i, YH) = area(ZH, k, j, i, YH) * q_val;
        integrand(k, j, i, ZH) = area(ZH, k, j, i, ZH) * q_val;
      }
    }
  }
  ops.integrate_and_diff_zhat(integrand, rhs);

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto scv = vol(k, j, i);
//        ThrowRequire(scv[0] > 0);
//        std::cout << "vol: " << scv << std::endl;

        integrand(k, j, i, XH) = -dqdx(k, j, i, XH) * scv;
        integrand(k, j, i, YH) = -dqdx(k, j, i, YH) * scv;
        integrand(k, j, i, ZH) = -dqdx(k, j, i, ZH) * scv;
      }
    }
  }
  ops.volume(integrand, rhs);


//  std::cout << "--------------------" << std::endl;
}

template <int poly_order, typename Scalar>
void green_gauss_rhs_linearized(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  const nodal_vector_view<poly_order, Scalar>& delta,
  nodal_vector_view<poly_order, Scalar>& rhs)
{
  static constexpr int n1D = poly_order + 1;

  nodal_vector_array<Scalar, poly_order> work_integrand;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto scv = vol(k, j, i);
        work_integrand(k, j, i, XH) = -delta(k, j, i, XH) * scv;
        work_integrand(k, j, i, YH) = -delta(k, j, i, YH) * scv;
        work_integrand(k, j, i, ZH) = -delta(k, j, i, ZH) * scv;
      }
    }
  }
  auto integrand = la::make_view(work_integrand);
  ops.volume(integrand, rhs);
}

template <int poly_order, typename Scalar>
void green_gauss_rhs_linearized(
  const CVFEMOperators<poly_order, Scalar>& ops,
  const nodal_scalar_view<poly_order, Scalar>& vol,
  const nodal_scalar_view<poly_order, Scalar>& delta,
  nodal_scalar_view<poly_order, Scalar>& rhs)
{
  static constexpr int n1D = poly_order + 1;

  nodal_scalar_array<Scalar, poly_order> work_integrand;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto scv = vol(k, j, i);
        work_integrand(k, j, i) = -delta(k, j, i) * scv;
      }
    }
  }
  auto integrand = la::make_view(work_integrand);
  ops.volume(integrand, rhs);
}

}
}
}

#endif
