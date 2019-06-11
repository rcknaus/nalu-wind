/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderScalarBDF2TimeDerivativeQuad_h
#define HighOrderScalarBDF2TimeDerivativeQuad_h

#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/DirectionMacros.h>
#include <CVFEMTypeDefs.h>

namespace sierra {
namespace nalu {
namespace tensor_assembly {

template <int p, typename Scalar>
void scalar_dt_lhs(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& vol,
  const double gamma1_div_dt,
  const nodal_scalar_view<p, Scalar>& rho_p1,
  matrix_view<p, Scalar>& lhs)
{
  constexpr int n1D = p + 1;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        auto rowIndex = idx<n1D>(n, m, l);
        for (int k = 0; k < n1D; ++k) {
          auto gammaWnk = gamma1_div_dt * weight(n, k);
          for (int j = 0; j < n1D; ++j) {
            auto gammWnkWmj = gammaWnk * weight(m, j);
            for (int i = 0; i < n1D; ++i) {
              lhs(rowIndex, idx<n1D>(k, j, i)) += gammWnkWmj * weight(l, i) * vol(k, j, i) * rho_p1(k, j, i);
            }
          }
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int p, typename Scalar>
void density_weighted_scalar_dt_rhs(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& metric,
  const double gamma_div_dt[3],
  const nodal_scalar_view<p, Scalar>& rhom1,
  const nodal_scalar_view<p, Scalar>& rhop0,
  const nodal_scalar_view<p, Scalar>& rhop1,
  const nodal_scalar_view<p, Scalar>& phim1,
  const nodal_scalar_view<p, Scalar>& phip0,
  const nodal_scalar_view<p, Scalar>& phip1,
  nodal_scalar_view<p, Scalar>& rhs)
{
  constexpr int n1D = p + 1;

  nodal_scalar_workview<p, Scalar> work_drhoqdt;
  auto& drhoqdt = work_drhoqdt.view();

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          drhoqdt(k, j, i) = -(
                gamma_div_dt[0] * rhop1(k, j, i) * phip1(k, j, i)
              + gamma_div_dt[1] * rhop0(k, j, i) * phip0(k, j, i)
              + gamma_div_dt[2] * rhom1(k, j, i) * phim1(k, j, i)
              ) * metric(k, j, i);
      }
    }
  }
  ops.volume(drhoqdt, rhs);
}
//--------------------------------------------------------------------------
template <int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION void scalar_dt_rhs(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& metric,
  const double gamma_div_dt[3],
  const nodal_scalar_view<p, Scalar>& qm1,
  const nodal_scalar_view<p, Scalar>& qp0,
  const nodal_scalar_view<p, Scalar>& qp1,
  nodal_scalar_view<p, Scalar>& rhs)
{
  constexpr int n1D = p + 1;

  nodal_scalar_array<Scalar, p> work_dqdt;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          work_dqdt(k, j, i) = -(
                gamma_div_dt[0] * qp1(k, j, i)
              + gamma_div_dt[1] * qp0(k, j, i)
              + gamma_div_dt[2] * qm1(k, j, i)
              ) * metric(k, j, i);
      }
    }
  }
  auto v_dqdt = la::make_view(work_dqdt);
  ops.volume(v_dqdt, rhs);
}

template <int p, typename Scalar>
void scalar_dt_rhs_linearized(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& volume,
  double gamma,
  const nodal_scalar_view<p, Scalar>& delta,
  nodal_scalar_view<p, Scalar>& rhs)
{
  constexpr int n1D = p + 1;
  nodal_scalar_array<Scalar, p> work_dqdt;
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          work_dqdt(k, j, i) = -delta(k, j, i) * volume(k, j, i) * gamma;
      }
    }
  }
  auto v_dqdt = la::make_view(work_dqdt);
  ops.volume(v_dqdt, rhs);
}


template <int p, typename Scalar>
void scalar_dt_lhs_diagonal(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& vol,  const double gamma1_div_dt,
  matrix_view<p, Scalar>& lhs)
{
  constexpr int n1D = p + 1;
  const auto& weight = ops.mat_.nodalWeights;

  for (int n = 0; n < n1D; ++n) {
    for (int m = 0; m < n1D; ++m) {
      for (int l = 0; l < n1D; ++l) {
        auto rowIndex = idx<n1D>(n, m, l);
        lhs(rowIndex, rowIndex) += gamma1_div_dt * weight(n, n) * weight(m,m) * weight(l,l) * vol(n, m, l);
      }
    }
  }
}

//--------------------------------------------------------------------------
template <int p, typename Scalar>
void density_dt_rhs(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_scalar_view<p, Scalar>& metric,
  const double gamma_div_dt[3],
  const nodal_scalar_view<p, Scalar>& rhom1,
  const nodal_scalar_view<p, Scalar>& rhop0,
  const nodal_scalar_view<p, Scalar>& rhop1,
  nodal_scalar_view<p, Scalar>& rhs)
{
  constexpr int n1D = p + 1;

  nodal_scalar_workview<p, Scalar> work_drhodt;
  auto& drhodt = work_drhodt.view();
  const double inv_projTimeScale = gamma_div_dt[0];

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
          drhodt(k, j, i) = -(
                gamma_div_dt[0] * rhop1(k, j, i)
              + gamma_div_dt[1] * rhop0(k, j, i)
              + gamma_div_dt[2] * rhom1(k, j, i)
              ) * metric(k, j, i) * inv_projTimeScale;
      }
    }
  }
  ops.volume(drhodt, rhs);
}

}
}
}

#endif
