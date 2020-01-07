#ifndef STK_SIMD_VALID_LENGTH_H
#define STK_SIMD_VALID_LENGTH_H

#include "Kokkos_View.hpp"
#include "stk_simd/Simd.hpp"

#include "matrix_free/KokkosFramework.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p, int len>
struct valid_offset_t
{
  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_elem_offset_view<p>& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, 0, 0, 0, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }

  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_face_offset_view<p>& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, 0, 0, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }

  static KOKKOS_FORCEINLINE_FUNCTION int
  valid_offset(int index, const const_node_offset_view& offsets)
  {
    int n = simd_len - 1;
    while (offsets(index, n) < 0 && n != 0) {
      --n;
    }
    return n + 1;
  }
};

template <int p>
struct valid_offset_t<p, 1>
{
  template <typename UnusedType>
  static KOKKOS_FORCEINLINE_FUNCTION int valid_offset(int, const UnusedType&)
  {
    return 1;
  }
};
} // namespace impl
template <int p>
KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_elem_offset_view<p>& offsets)
{
  return impl::valid_offset_t<p, simd_len>::valid_offset(index, offsets);
}

template <int p>
KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_face_offset_view<p>& offsets)
{
  return impl::valid_offset_t<p, simd_len>::valid_offset(index, offsets);
}

KOKKOS_FORCEINLINE_FUNCTION int
valid_offset(int index, const const_node_offset_view& offsets)
{
  return impl::valid_offset_t<0, simd_len>::valid_offset(index, offsets);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif