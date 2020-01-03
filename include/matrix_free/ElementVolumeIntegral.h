#ifndef ELEMENT_VOLUME_INTEGRAL_H
#define ELEMENT_VOLUME_INTEGRAL_H

#include <cmath>

#include "matrix_free/Coefficients.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

#define XH 0
#define YH 1
#define ZH 2

template <int p, typename InArray, typename ScratchArray, typename OutArray>
KOKKOS_FUNCTION void
volume(
  const InArray& in,
  const typename Coeffs<p>::nodal_matrix_type& vandermonde,
  ScratchArray& scratch_1,
  ScratchArray& scratch_2,
  OutArray& out)
{
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(i, q) * in(k, j, q);
        }
        scratch_1(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(j, q) * scratch_1(k, q, i);
        }
        scratch_2(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(k, q) * scratch_2(q, j, i);
        }
        out(k, j, i) = acc;
      }
    }
  }
}

template <
  int p,
  typename VolumeArray,
  typename InArray,
  typename ScratchArray,
  typename OutArray>
KOKKOS_FUNCTION void
consistent_mass_time_derivative(
  int index,
  Kokkos::Array<double, 3> gammas,
  const VolumeArray& vol,
  const InArray& qm1,
  const InArray& qp0,
  const InArray& qp1,
  ScratchArray& scratch,
  OutArray& out)
{
  static constexpr auto vandermonde = Coeffs<p>::W;

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc +=
            vandermonde(i, q) * -vol(index, k, j, i) *
            (gammas[0] * qp1(index, k, j, i) + gammas[1] * qp0(index, k, j, i) +
             gammas[2] * qm1(index, k, j, i));
        }
        out(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(j, q) * out(k, q, i);
        }
        scratch(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(k, q) * scratch(q, j, i);
        }
        out(k, j, i) = acc;
      }
    }
  }
}

template <int p, typename VolumeArray, typename InArray, typename OutArray>
KOKKOS_FUNCTION void
lumped_time_derivative(
  int index,
  Kokkos::Array<double, 3> gammas,
  const VolumeArray& vol,
  const InArray& qm1,
  const InArray& qp0,
  const InArray& qp1,
  OutArray& out)
{
  static constexpr auto lumped = Coeffs<p>::Wl;
  for (int k = 0; k < p + 1; ++k) {
    const auto Wk = lumped[k];
    for (int j = 0; j < p + 1; ++j) {
      const auto WkWj = Wk * lumped[j];
      for (int i = 0; i < p + 1; ++i) {
        out(k, j, i) =
          -WkWj * lumped[i] * vol(index, k, j, i) *
          (gammas[0] * qp1(index, k, j, i) + gammas[1] * qp0(index, k, j, i) +
           gammas[2] * qm1(index, k, j, i));
      }
    }
  }
}

template <
  int p,
  typename VolumeArray,
  typename DeltaArray,
  typename ScratchArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
mass_term(
  int index,
  double gamma,
  const VolumeArray& volume_metric,
  const DeltaArray& delta,
  ScratchArray& scratch,
  OutArray& out)
{
  static constexpr auto vandermonde = Coeffs<p>::W;

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc +=
            vandermonde(i, q) * volume_metric(index, k, j, q) * delta(k, j, q);
        }
        out(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(j, q) * out(k, q, i);
        }
        scratch(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(k, q) * scratch(q, j, i);
        }
        out(k, j, i) = -gamma * acc;
      }
    }
  }
}

template <
  int p,
  typename VolumeArray,
  typename DeltaArray,
  typename ScratchArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
lumped_mass_term(
  int index,
  double gamma,
  const VolumeArray& volume_metric,
  const DeltaArray& delta,
  ScratchArray& scratch,
  OutArray& out)
{
  static constexpr auto vandermonde = Coeffs<p>::W;

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc +=
            vandermonde(i, q) * volume_metric(index, k, j, q) * delta(k, j, q);
        }
        out(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc = 0;
        ;
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(j, q) * out(k, q, i);
        }
        scratch(k, j, i) = acc;
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(k, q) * scratch(q, j, i);
        }
        out(k, j, i) = -gamma * acc;
      }
    }
  }
}

#undef XH
#undef YH
#undef ZH

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
