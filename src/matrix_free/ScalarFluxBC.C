#include "matrix_free/ScalarFluxBC.h"

#include "matrix_free/ConductionInterior.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosFramework.h"

#include "Kokkos_Macros.hpp"
#include "Teuchos_RCP.hpp"
#include "stk_simd/Simd.hpp"
#include "Tpetra_Operator.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace {
template <
  int p,
  typename CoeffArray,
  typename ScratchArray,
  typename FaceRankInput,
  typename FaceRankOutput>
KOKKOS_FUNCTION void
integrate_face(
  const CoeffArray& vandermonde,
  ScratchArray& scratch,
  const FaceRankInput& in,
  FaceRankOutput& out)
{
  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      ftype acc = 0;
      for (int q = 0; q < p + 1; ++q) {
        acc += vandermonde(i, q) * in(j, q);
      }
      scratch(j, i) = acc;
    }
  }

  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      out(j, i) = 0;
    }

    for (int q = 0; q < p + 1; ++q) {
      const auto temp = vandermonde(j, q);
      for (int i = 0; i < p + 1; ++i) {
        out(j, i) += temp * scratch(q, i);
      }
    }
  }
}

} // namespace

namespace impl {
template <int p>
void
scalar_neumann_residual_t<p>::invoke(
  const_face_offset_view<p> offsets,
  const_face_scalar_view<p> dqdn,
  const_face_vector_view<p> areav,
  tpetra_view_type owned_rhs)
{
  ngp::ProfilingBlock pf("scalar_neumann_residual");

  constexpr auto vandermonde = Coeffs<p>::W;
  Kokkos::parallel_for(
    "flux_residual", offsets.extent_int(0), KOKKOS_LAMBDA(int index) {
      LocalArray<ftype[p + 1][p + 1]> flux;
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          const auto amag = stk::math::sqrt(
            areav(index, j, i, 0) * areav(index, j, i, 0) +
            areav(index, j, i, 1) * areav(index, j, i, 1) +
            areav(index, j, i, 2) * areav(index, j, i, 2));
          flux(j, i) = dqdn(index, j, i) * amag;
        }
      }

      LocalArray<ftype[p + 1][p + 1]> element_rhs;
      {
        LocalArray<ftype[p + 1][p + 1]> scratch;
        integrate_face<p>(vandermonde, scratch, flux, element_rhs);
      }

      const int valid_length = valid_offset<p>(index, offsets);
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          for (int n = 0; n < valid_length; ++n) {
            Kokkos::atomic_add(
              &owned_rhs(offsets(index, j, i, n), 0),
              stk::simd::get_data(element_rhs(j, i), n));
          }
        }
      }
    });
}
INSTANTIATE_POLYSTRUCT(scalar_neumann_residual_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
