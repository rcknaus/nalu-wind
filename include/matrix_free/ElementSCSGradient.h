#ifndef ELEMENT_SCS_GRADIENT_H
#define ELEMENT_SCS_GRADIENT_H

#include <Kokkos_MemoryTraits.hpp>
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
KOKKOS_FORCEINLINE_FUNCTION void
grad_scs(
  const InArray& in,
  const typename Coeffs<p>::scs_matrix_type& flux_point_interpolant,
  const typename Coeffs<p>::scs_matrix_type& flux_point_derivative,
  const typename Coeffs<p>::nodal_matrix_type& nodal_derivative,
  ScratchArray& interp,
  OutArray& grad)
{
  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        interp(XH, l, s, r) = 0;
      }
      for (int r = 0; r < p + 1; ++r) {
        for (int q = 0; q < p + 1; ++q) {
          interp(XH, l, s, r) += flux_point_interpolant(l, q) * in(s, r, q);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        interp(YH, l, s, r) = 0;
      }
      for (int q = 0; q < p + 1; ++q) {
        const auto temp = flux_point_interpolant(l, q);
        for (int r = 0; r < p + 1; ++r) {
          interp(YH, l, s, r) += temp * in(s, q, r);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int r = 0; r < p + 1; ++r) {
      for (int s = 0; s < p + 1; ++s) {
        interp(ZH, l, s, r) = 0;
      }
    }
    for (int q = 0; q < p + 1; ++q) {
      const auto temp = flux_point_interpolant(l, q);
      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          interp(ZH, l, s, r) += temp * in(q, s, r);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        grad(XH, l, s, r, XH) = 0;
      }
      for (int r = 0; r < p + 1; ++r) {
        for (int q = 0; q < p + 1; ++q) {
          grad(XH, l, s, r, XH) += flux_point_derivative(l, q) * in(s, r, q);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        grad(YH, l, s, r, YH) = 0;
      }
      for (int q = 0; q < p + 1; ++q) {
        const auto temp = flux_point_derivative(l, q);
        for (int r = 0; r < p + 1; ++r) {
          grad(YH, l, s, r, YH) += temp * in(s, q, r);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int r = 0; r < p + 1; ++r) {
      for (int s = 0; s < p + 1; ++s) {
        grad(ZH, l, s, r, ZH) = 0;
      }
    }
    for (int q = 0; q < p + 1; ++q) {
      const auto temp = flux_point_derivative(l, q);
      for (int s = 0; s < p + 1; ++s) {
        for (int r = 0; r < p + 1; ++r) {
          grad(ZH, l, s, r, ZH) += temp * in(q, s, r);
        }
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(r, q) * interp(XH, l, s, q);
        }
        grad(XH, l, s, r, YH) = acc;
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(s, q) * interp(XH, l, q, r);
        }
        grad(XH, l, s, r, ZH) = acc;
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(r, q) * interp(YH, l, s, q);
        }
        grad(YH, l, s, r, XH) = acc;
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(s, q) * interp(YH, l, q, r);
        }
        grad(YH, l, s, r, ZH) = acc;
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(r, q) * interp(ZH, l, s, q);
        }
        grad(ZH, l, s, r, XH) = acc;
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        ftype acc = 0;
        for (int q = 0; q < p + 1; ++q) {
          acc += nodal_derivative(s, q) * interp(ZH, l, q, r);
        }
        grad(ZH, l, s, r, YH) = acc;
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
