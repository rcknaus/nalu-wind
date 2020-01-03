#include "matrix_free/ConductionDiagonal.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <stk_simd/Simd.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {
namespace {

#define XH 0
#define YH 1
#define ZH 2
template <
  int p,
  typename NodalType,
  typename FluxType,
  typename MetricType,
  typename LHSType>
KOKKOS_FUNCTION void
conduction_xh(
  int index,
  const NodalType& vandermonde,
  const NodalType& nodal_derivative,
  const FluxType& flux_point_derivative,
  const FluxType& flux_point_interpolant,
  const MetricType& metric,
  LHSType& lhs)
{
  constexpr int n1D = p + 1;
  for (int n = 0; n < n1D; ++n) {
    const ftype Wn = vandermonde(n, n);
    for (int m = 0; m < n1D; ++m) {
      const ftype Wm = vandermonde(m, m);
      const ftype WnWm = Wn * Wm;
      {
        constexpr int l_minus = 0;
        const ftype orth = WnWm * metric(index, XH, l_minus, n, m, 0);
        ftype non_orth_y = 0.0;
        ftype non_orth_z = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_y += Wn * vandermonde(m, q) * nodal_derivative(q, m) *
                        metric(index, XH, l_minus, n, q, 1);
          non_orth_z += vandermonde(n, q) * Wm * nodal_derivative(q, n) *
                        metric(index, XH, l_minus, q, m, 2);
        }

        lhs(n, m, l_minus) +=
          orth * flux_point_derivative(l_minus, l_minus) +
          flux_point_interpolant(l_minus, l_minus) * (non_orth_y + non_orth_z);
      }

      {
        for (int l = 1; l < n1D - 1; ++l) {
          const ftype orthm1 = WnWm * metric(index, XH, l - 1, n, m, 0);
          const ftype orthp0 = WnWm * metric(index, XH, l + 0, n, m, 0);
          ftype non_orth_y_m1 = 0.0;
          ftype non_orth_y_p0 = 0.0;
          ftype non_orth_z_m1 = 0.0;
          ftype non_orth_z_p0 = 0.0;

          for (int q = 0; q < n1D; ++q) {
            const ftype wy = Wn * vandermonde(m, q) * nodal_derivative(q, m);
            non_orth_y_m1 += wy * metric(index, XH, l - 1, n, q, 1);
            non_orth_y_p0 += wy * metric(index, XH, l + 0, n, q, 1);

            const ftype wz = vandermonde(n, q) * Wm * nodal_derivative(q, n);
            non_orth_z_m1 += wz * metric(index, XH, l - 1, q, m, 2);
            non_orth_z_p0 += wz * metric(index, XH, l + 0, q, m, 2);
          }

          const ftype integrated_flux_m =
            orthm1 * flux_point_derivative(l - 1, l) +
            flux_point_interpolant(l - 1, l) * (non_orth_y_m1 + non_orth_z_m1);
          const ftype integrated_flux_p =
            orthp0 * flux_point_derivative(l + 0, l) +
            flux_point_interpolant(l + 0, l) * (non_orth_y_p0 + non_orth_z_p0);
          lhs(n, m, l) += integrated_flux_p - integrated_flux_m;
        }
      }

      {
        constexpr int l_plus = n1D - 1;
        const ftype orth = WnWm * metric(index, XH, l_plus - 1, n, m, 0);
        ftype non_orth_y = 0.0;
        ftype non_orth_z = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_y += Wn * vandermonde(m, q) * nodal_derivative(q, m) *
                        metric(index, XH, l_plus - 1, n, q, 1);
          non_orth_z += vandermonde(n, q) * Wm * nodal_derivative(q, n) *
                        metric(index, XH, l_plus - 1, q, m, 2);
        }

        lhs(n, m, l_plus) -= orth * flux_point_derivative(l_plus - 1, l_plus) +
                             flux_point_interpolant(l_plus - 1, l_plus) *
                               (non_orth_y + non_orth_z);
      }
    }
  }
}

template <
  int p,
  typename NodalType,
  typename FluxType,
  typename MetricType,
  typename LHSType>
KOKKOS_FUNCTION void
conduction_yh(
  int index,
  const NodalType& vandermonde,
  const NodalType& nodal_derivative,
  const FluxType& flux_point_derivative,
  const FluxType& flux_point_interpolant,
  const MetricType& metric,
  LHSType& lhs)
{
  constexpr int n1D = p + 1;
  for (int n = 0; n < n1D; ++n) {
    const ftype Wn = vandermonde(n, n);
    for (int l = 0; l < n1D; ++l) {
      const ftype Wl = vandermonde(l, l);
      const ftype WnWl = Wn * Wl;
      {
        constexpr int m_minus = 0;
        const ftype orth = WnWl * metric(index, YH, m_minus, n, l, 0);

        ftype non_orth_x = 0.0;
        ftype non_orth_z = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_x += Wn * vandermonde(l, q) * nodal_derivative(q, l) *
                        metric(index, YH, m_minus, n, q, 1);
          non_orth_z += vandermonde(n, q) * Wl * nodal_derivative(q, n) *
                        metric(index, YH, m_minus, q, l, 2);
        }

        lhs(n, m_minus, l) +=
          orth * flux_point_derivative(m_minus, m_minus) +
          flux_point_interpolant(m_minus, m_minus) * (non_orth_x + non_orth_z);
      }

      {
        for (int m = 1; m < n1D - 1; ++m) {
          const ftype orthm1 = WnWl * metric(index, YH, m - 1, n, l, 0);
          const ftype orthp0 = WnWl * metric(index, YH, m + 0, n, l, 0);
          ftype non_orth_x_m1 = 0.0;
          ftype non_orth_x_p0 = 0.0;
          ftype non_orth_z_m1 = 0.0;
          ftype non_orth_z_p0 = 0.0;

          for (int q = 0; q < n1D; ++q) {
            const ftype wx = Wn * vandermonde(l, q) * nodal_derivative(q, l);
            non_orth_x_m1 += wx * metric(index, YH, m - 1, n, q, 1);
            non_orth_x_p0 += wx * metric(index, YH, m + 0, n, q, 1);

            const ftype wz = vandermonde(n, q) * Wl * nodal_derivative(q, n);
            non_orth_z_m1 += wz * metric(index, YH, m - 1, q, l, 2);
            non_orth_z_p0 += wz * metric(index, YH, m + 0, q, l, 2);
          }

          const ftype integrated_flux_m =
            orthm1 * flux_point_derivative(m - 1, m) +
            flux_point_interpolant(m - 1, m) * (non_orth_x_m1 + non_orth_z_m1);
          const ftype integrated_flux_p =
            orthp0 * flux_point_derivative(m + 0, m) +
            flux_point_interpolant(m + 0, m) * (non_orth_x_p0 + non_orth_z_p0);
          lhs(n, m, l) += integrated_flux_p - integrated_flux_m;
        }
      }

      {
        constexpr int m_plus = n1D - 1;
        const ftype orth = WnWl * metric(index, YH, m_plus - 1, n, l, 0);

        ftype non_orth_x = 0.0;
        ftype non_orth_z = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_x += Wn * vandermonde(l, q) * nodal_derivative(q, l) *
                        metric(index, YH, m_plus - 1, n, q, 1);
          non_orth_z += vandermonde(n, q) * Wl * nodal_derivative(q, n) *
                        metric(index, YH, m_plus - 1, q, l, 2);
        }

        lhs(n, m_plus, l) -= orth * flux_point_derivative(m_plus - 1, m_plus) +
                             flux_point_interpolant(m_plus - 1, m_plus) *
                               (non_orth_x + non_orth_z);
      }
    }
  }
}

template <
  int p,
  typename NodalType,
  typename FluxType,
  typename MetricType,
  typename LHSType>
KOKKOS_FUNCTION void
conduction_zh(
  int index,
  const NodalType& vandermonde,
  const NodalType& nodal_derivative,
  const FluxType& flux_point_derivative,
  const FluxType& flux_point_interpolant,
  const MetricType& metric,
  LHSType& lhs)
{
  const int n1D = p + 1;
  for (int m = 0; m < n1D; ++m) {
    const ftype Wm = vandermonde(m, m);
    for (int l = 0; l < n1D; ++l) {
      const ftype Wl = vandermonde(l, l);
      const ftype WmWl = Wm * Wl;

      {
        constexpr int n_minus = 0;
        const ftype orth = WmWl * metric(index, ZH, n_minus, m, l, 0);
        ftype non_orth_x = 0.0;
        ftype non_orth_y = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_x += Wm * vandermonde(l, q) * nodal_derivative(q, l) *
                        metric(index, ZH, n_minus, m, q, 1);
          non_orth_y += vandermonde(m, q) * Wl * nodal_derivative(q, m) *
                        metric(index, ZH, n_minus, q, l, 2);
        }

        lhs(n_minus, m, l) +=
          orth * flux_point_derivative(n_minus, n_minus) +
          flux_point_interpolant(n_minus, n_minus) * (non_orth_x + non_orth_y);
      }

      {
        for (int n = 1; n < n1D - 1; ++n) {
          const ftype orthm1 = WmWl * metric(index, ZH, n - 1, m, l, 0);
          const ftype orthp0 = WmWl * metric(index, ZH, n + 0, m, l, 0);

          ftype non_orth_x_m1 = 0.0;
          ftype non_orth_x_p0 = 0.0;
          ftype non_orth_y_m1 = 0.0;
          ftype non_orth_y_p0 = 0.0;

          for (int q = 0; q < n1D; ++q) {
            const ftype wx = Wm * vandermonde(l, q) * nodal_derivative(q, l);
            non_orth_x_m1 += wx * metric(index, ZH, n - 1, m, q, 1);
            non_orth_x_p0 += wx * metric(index, ZH, n + 0, m, q, 1);

            const ftype wy = vandermonde(m, q) * Wl * nodal_derivative(q, m);
            non_orth_y_m1 += wy * metric(index, ZH, n - 1, q, l, 2);
            non_orth_y_p0 += wy * metric(index, ZH, n + 0, q, l, 2);
          }

          const ftype integrated_flux_m =
            orthm1 * flux_point_derivative(n - 1, n) +
            flux_point_interpolant(n - 1, n) * (non_orth_x_m1 + non_orth_y_m1);
          const ftype integrated_flux_p =
            orthp0 * flux_point_derivative(n + 0, n) +
            flux_point_interpolant(n + 0, n) * (non_orth_x_p0 + non_orth_y_p0);

          lhs(n, m, l) += integrated_flux_p - integrated_flux_m;
        }
      }

      {
        constexpr int n_plus = n1D - 1;

        const ftype orth = WmWl * metric(index, ZH, n_plus - 1, m, l, 0);

        ftype non_orth_x = 0.0;
        ftype non_orth_y = 0.0;
        for (int q = 0; q < n1D; ++q) {
          non_orth_x += Wm * vandermonde(l, q) * nodal_derivative(q, l) *
                        metric(index, ZH, n_plus - 1, m, q, 1);
          non_orth_y += vandermonde(m, q) * Wl * nodal_derivative(q, m) *
                        metric(index, ZH, n_plus - 1, q, l, 2);
        }

        lhs(n_plus, m, l) -= orth * flux_point_derivative(n_plus - 1, n_plus) +
                             flux_point_interpolant(n_plus - 1, n_plus) *
                               (non_orth_x + non_orth_y);
      }
    }
  }
}

#undef XH
#undef YH
#undef ZH
} // namespace

template <int p>
void
conduction_diagonal_t<p>::invoke(
  double gamma,
  const_elem_offset_view<p> offsets,
  const_scalar_view<p> volumes,
  const_scs_vector_view<p> metric,
  tpetra_view_type owned_yout)
{
  constexpr auto flux_point_interpolant = Coeffs<p>::Nt;
  constexpr auto flux_point_derivative = Coeffs<p>::Dt;
  constexpr auto nodal_derivative = Coeffs<p>::D;
  constexpr auto vandermonde = Coeffs<p>::W;
  constexpr auto Wl = Coeffs<p>::Wl;

  Kokkos::parallel_for(
    "diagonal", offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
      constexpr int n1D = p + 1;
      LocalArray<ftype[n1D][n1D][n1D]> lhs;
      for (int k = 0; k < p + 1; ++k) {
        const auto gammaWk = gamma * Wl(k);
        for (int j = 0; j < p + 1; ++j) {
          const auto gammaWkWj = gammaWk * Wl(j);
          for (int i = 0; i < p + 1; ++i) {
            lhs(k, j, i) = gammaWkWj * Wl(i) * volumes(index, k, j, i);
          }
        }
      }

      conduction_xh<p>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);
      conduction_yh<p>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);
      conduction_zh<p>(
        index, vandermonde, nodal_derivative, flux_point_derivative,
        flux_point_interpolant, metric, lhs);

      const int valid_simd_len = valid_offset<p>(index, offsets);
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_simd_len; ++n) {
              Kokkos::atomic_add(
                &owned_yout(offsets(index, k, j, i, n), 0),
                stk::simd::get_data(lhs(k, j, i), n));
            }
          }
        }
      }
    });
}
INSTANTIATE_POLYSTRUCT(conduction_diagonal_t);
} // namespace impl

void
dirichlet_diagonal(
  const_node_offset_view offsets,
  int max_owned_lid,
  tpetra_view_type owned_yout)
{
  Kokkos::parallel_for(
    "dirichlet_diagonal", offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
      const int valid_simd_len = valid_offset(index, offsets);
      for (int n = 0; n < valid_simd_len; ++n) {
        const auto row_lid = offsets(index, n);
        owned_yout(row_lid, 0) = (row_lid < max_owned_lid);
      }
    });
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
