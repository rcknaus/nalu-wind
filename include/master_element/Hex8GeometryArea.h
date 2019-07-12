/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef Hex8GeometryArea_h
#define Hex8GeometryArea_h

#include <AlgTraits.h>

#include <master_element/TensorOps.h>
#include <CVFEMTypeDefs.h>

#include <SimdInterface.h>
#include <Kokkos_Core.hpp>

#include <stk_util/util/ReportHandler.hpp>

#include <master_element/DirectionMacros.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <array>
#include <type_traits>

namespace sierra {
namespace nalu {

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_xh(int d, const Scalar base_box[3][8], const Scalar interpj[2], const Scalar interpk[2])
{
  return( -interpj[0] * interpk[0] * base_box[d][0]
        +  interpj[0] * interpk[0] * base_box[d][1]
        +  interpj[1] * interpk[0] * base_box[d][2]
        -  interpj[1] * interpk[0] * base_box[d][3]
        -  interpj[0] * interpk[1] * base_box[d][4]
        +  interpj[0] * interpk[1] * base_box[d][5]
        +  interpj[1] * interpk[1] * base_box[d][6]
        -  interpj[1] * interpk[1] * base_box[d][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_yh(int d, const Scalar base_box[3][8],  const Scalar interpi[2], const Scalar interpk[2])
{
  return ( -interpi[0] * interpk[0] * base_box[d][0]
         -  interpi[1] * interpk[0] * base_box[d][1]
         +  interpi[1] * interpk[0] * base_box[d][2]
         +  interpi[0] * interpk[0] * base_box[d][3]
         -  interpi[0] * interpk[1] * base_box[d][4]
         -  interpi[1] * interpk[1] * base_box[d][5]
         +  interpi[1] * interpk[1] * base_box[d][6]
         +  interpi[0] * interpk[1] * base_box[d][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
Scalar jacobian_component_zh(int d, const Scalar base_box[3][8], const Scalar interpi[2], const Scalar interpj[2])
{
  return ( -interpi[0] * interpj[0] * base_box[d][0]
         -  interpi[1] * interpj[0] * base_box[d][1]
         -  interpi[1] * interpj[1] * base_box[d][2]
         -  interpi[0] * interpj[1] * base_box[d][3]
         +  interpi[0] * interpj[0] * base_box[d][4]
         +  interpi[1] * interpj[0] * base_box[d][5]
         +  interpi[1] * interpj[1] * base_box[d][6]
         +  interpi[0] * interpj[1] * base_box[d][7]
  ) * 0.5;
}

template <int di, int dj, typename Scalar>
Scalar hex_jacobian_component(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2])
{
  return (dj == XH) ?
  ( -interpj[0] * interpk[0] * base_box[di][0]
  +  interpj[0] * interpk[0] * base_box[di][1]
  +  interpj[1] * interpk[0] * base_box[di][2]
  -  interpj[1] * interpk[0] * base_box[di][3]
  -  interpj[0] * interpk[1] * base_box[di][4]
  +  interpj[0] * interpk[1] * base_box[di][5]
  +  interpj[1] * interpk[1] * base_box[di][6]
  -  interpj[1] * interpk[1] * base_box[di][7]
  ) * 0.5
  : (dj == YH) ?
  ( -interpi[0] * interpk[0] * base_box[di][0]
  -  interpi[1] * interpk[0] * base_box[di][1]
  +  interpi[1] * interpk[0] * base_box[di][2]
  +  interpi[0] * interpk[0] * base_box[di][3]
  -  interpi[0] * interpk[1] * base_box[di][4]
  -  interpi[1] * interpk[1] * base_box[di][5]
  +  interpi[1] * interpk[1] * base_box[di][6]
  +  interpi[0] * interpk[1] * base_box[di][7]
  ) * 0.5
  :
  ( -interpi[0] * interpj[0] * base_box[di][0]
  -  interpi[1] * interpj[0] * base_box[di][1]
  -  interpi[1] * interpj[1] * base_box[di][2]
  -  interpi[0] * interpj[1] * base_box[di][3]
  +  interpi[0] * interpj[0] * base_box[di][4]
  +  interpi[1] * interpj[0] * base_box[di][5]
  +  interpi[1] * interpj[1] * base_box[di][6]
  +  interpi[0] * interpj[1] * base_box[di][7]
  ) * 0.5;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_jacobian(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar jac[3][3])
{
  for (int d = 0; d < 3; ++d) {
    jac[0][d] = jacobian_component_xh(d, base_box, interpj, interpk);
  }

  for (int d = 0; d < 3; ++d) {
    jac[1][d] = jacobian_component_yh(d, base_box, interpi, interpk);
  }

  for (int d = 0; d < 3; ++d) {
    jac[2][d] = jacobian_component_zh(d, base_box, interpi, interpj);
  }
}


template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_jacobian_t(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar jac[3][3])
{
  for (int d = 0; d < 3; ++d) {
    jac[d][0] = jacobian_component_xh(d, base_box, interpj, interpk);
    jac[d][1] = jacobian_component_yh(d, base_box, interpi, interpk);
    jac[d][2] = jacobian_component_zh(d, base_box, interpi, interpj);
  }
}

template<int dir, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void areav_from_jacobian_t(
  const Scalar jact[3][3],
  Scalar area[3])
{
  constexpr int orth_comp_1 = (dir == XH) ? ZH : (dir == YH) ? XH : YH;
  constexpr int orth_comp_2 = (dir == XH) ? YH : (dir == YH) ? ZH : XH;
  area[XH] = jact[YH][orth_comp_1] * jact[ZH][orth_comp_2] - jact[ZH][orth_comp_1] * jact[YH][orth_comp_2];
  area[YH] = jact[ZH][orth_comp_1] * jact[XH][orth_comp_2] - jact[XH][orth_comp_1] * jact[ZH][orth_comp_2];
  area[ZH] = jact[XH][orth_comp_1] * jact[YH][orth_comp_2] - jact[YH][orth_comp_1] * jact[XH][orth_comp_2];
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_x(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{
//  const auto dx_ds1 = hex_jacobian_component<XH, ZH>(base_box,interpi, interpj, interpk);

  NALU_ALIGNED Scalar jac[3][3];
  hex_jacobian_t(base_box, interpi, interpj, interpk, jac);
  areav_from_jacobian_t<XH>(jac, area);

//
//
//  const auto dx_ds1 = jacobian_component_zh(0, base_box, interpi, interpj);
//  const auto dx_ds2 = jacobian_component_yh(0, base_box, interpi, interpk);
//
//  const auto dy_ds1 = jacobian_component_zh(1, base_box, interpi, interpj);
//  const auto dy_ds2 = jacobian_component_yh(1, base_box, interpi, interpk);
//
//  const auto dz_ds1 = jacobian_component_zh(2, base_box, interpi, interpj);
//  const auto dz_ds2 = jacobian_component_yh(2, base_box, interpi, interpk);
//
//  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
//  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
//  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}


template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_y(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{

  NALU_ALIGNED Scalar jac[3][3];
  hex_jacobian_t(base_box, interpi, interpj, interpk, jac);
  areav_from_jacobian_t<YH>(jac, area);
//
//  const auto dx_ds1 = jacobian_component_xh(0, base_box, interpj, interpk);
//  const auto dx_ds2 = jacobian_component_zh(0, base_box, interpi, interpj);
//
//  const auto dy_ds1 = jacobian_component_xh(1, base_box, interpj, interpk);
//  const auto dy_ds2 = jacobian_component_zh(1, base_box, interpi, interpj);
//
//  const auto dz_ds1 = jacobian_component_xh(2, base_box, interpj, interpk);
//  const auto dz_ds2 = jacobian_component_zh(2, base_box, interpi, interpj);
//
//  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
//  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
//  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

template<typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_areav_z(
  const Scalar base_box[3][8],
  const Scalar interpi[2],
  const Scalar interpj[2],
  const Scalar interpk[2],
  Scalar area[3])
{
  NALU_ALIGNED Scalar jac[3][3];
  hex_jacobian_t(base_box, interpi, interpj, interpk, jac);
  areav_from_jacobian_t<ZH>(jac, area);
//
//  const auto dx_ds1 = jacobian_component_yh(0, base_box, interpi, interpk);
//  const auto dx_ds2 = jacobian_component_xh(0, base_box, interpj, interpk);
//
//  const auto dy_ds1 = jacobian_component_yh(1, base_box, interpi, interpk);
//  const auto dy_ds2 = jacobian_component_xh(1, base_box, interpj, interpk);
//
//  const auto dz_ds1 = jacobian_component_yh(2, base_box, interpi, interpk);
//  const auto dz_ds2 = jacobian_component_xh(2, base_box, interpj, interpk);
//
//  area[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
//  area[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
//  area[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}

template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void hex_vertex_coordinates(
  const nodal_vector_view<p, Scalar>& xc,
  Scalar base_box[3][8])
{
  for (int d = 0; d < 3; ++d) {
    base_box[d][0] = xc(0, 0, 0, d);
    base_box[d][1] = xc(0, 0, p, d);
    base_box[d][2] = xc(0, p, p, d);
    base_box[d][3] = xc(0, p, 0, d);

    base_box[d][4] = xc(p, 0, 0, d);
    base_box[d][5] = xc(p, 0, p, d);
    base_box[d][6] = xc(p, p, p, d);
    base_box[d][7] = xc(p, p, 0, d);
  }
}

template <int p> void SparsifiedLaplacianInterior<p>::compute_lhs_simd()
{
  std::vector<stk::mesh::Entity> entities(2, stk::mesh::Entity());
  std::vector<int> scratchIds(2, 0);
  std::vector<double> scratchVals(2, 0.0);

  std::vector<double> rhs(2, 0.0);
  std::vector<double> lhs(4, 0.0);
  static constexpr int perm[2][2][2] = { {0, 1}, {3, 2}, {4, 5}, {7, 6}};

  static constexpr LocalArray<int[2][2][2]> perm = {{
      {0, 1}, {3, 2}, {4, 5}, {7, 6}
  }};

//  Kokkos::

  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    auto elem_coords = nodal_vector_view<p>(&coords_(index,0,0,0,0));
    auto all_lhs = sparsified_laplacian_lhs<p>(elem_coords);
    for (int nsimd= 0; nsimd < simdLen && entities_(index, nsimd, 0,0,0).is_local_offset_valid(); ++nsimd) {
      for (int n = 0; n < p; ++n) {
        for (int m = 0; m < p; ++m) {
          for (int l = 0; l < p; ++l) {
            // edge ordinal 0
            entities[0] = entities_(index, nsimd, n+0,m+0,l+0);
            entities[1] = entities_(index, nsimd, n+0,m+0,l+1);
            lhs[0] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,0), perm(0,0,0)), nsimd);
            lhs[1] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,0), perm(0,0,1)), nsimd);
            lhs[2] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,1), perm(0,0,0)), nsimd);
            lhs[3] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,1), perm(0,0,1)), nsimd);
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);

            // edge ordinal 1
            entities[0] = entities_(index, nsimd, n+0,m+0,l+1);
            entities[1] = entities_(index, nsimd, n+0,m+1,l+1);
            lhs[0] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,1), perm(0,0,1)), nsimd);
            lhs[1] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,1), perm(0,1,1)), nsimd);
            lhs[2] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,1), perm(0,0,1)), nsimd);
            lhs[3] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,1), perm(0,1,1)), nsimd);
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);

            // edge ordinal 2
            entities[0] = entities_(index, nsimd, n+0,m+1,l+1);
            entities[1] = entities_(index, nsimd, n+0,m+1,l+0);
            lhs[0] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,1), perm(0,1,1)), nsimd);
            lhs[1] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,1), perm(0,1,0)), nsimd);
            lhs[2] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,0), perm(0,1,1)), nsimd);
            lhs[3] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,0), perm(0,1,0)), nsimd);
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);

            // edge ordinal 3
            entities[0] = entities_(index, nsimd, n+0,m+1,l+0);
            entities[1] = entities_(index, nsimd, n+0,m+0,l+0);
            lhs[0] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,0), perm(0,0,0)), nsimd);
            lhs[1] = stk::simd::get_data(all_lhs(n, m, l, perm(0,1,0), perm(0,1,0)), nsimd);
            lhs[2] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,0), perm(0,1,1)), nsimd);
            lhs[3] = stk::simd::get_data(all_lhs(n, m, l, perm(0,0,0), perm(0,0,0)), nsimd);
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);

            // edge ordinal 4
            entities[0] = entities_(index, nsimd, n+1,m+0,l+0);
            entities[1] = entities_(index, nsimd, n+1,m+0,l+1);

            // edge ordinal 5
            entities[0] = entities_(index, nsimd, n+1,m+0,l+1);
            entities[1] = entities_(index, nsimd, n+1,m+1,l+1);

            // edge ordinal 6
            entities[0] = entities_(index, nsimd, n+1,m+1,l+1);
            entities[1] = entities_(index, nsimd, n+1,m+1,l+0);

            // edge ordinal 7
            entities[0] = entities_(index, nsimd, n+1,m+1,l+0);
            entities[1] = entities_(index, nsimd, n+1,m+0,l+0);

            // edge ordinal 8
            entities[0] = entities_(index, nsimd, n+0,m+0,l+0);
            entities[1] = entities_(index, nsimd, n+1,m+0,l+0);

            // edge ordinal 9
            entities[0] = entities_(index, nsimd, n+0,m+0,l+1);
            entities[1] = entities_(index, nsimd, n+1,m+0,l+1);

            // edge ordinal 10
            entities[0] = entities_(index, nsimd, n+0,m+1,l+1);
            entities[1] = entities_(index, nsimd, n+1,m+1,l+1);

            // edge ordinal 11
            entities[0] = entities_(index, nsimd, n+0,m+1,l+0);
            entities[1] = entities_(index, nsimd, n+1,m+1,l+0);



            entities[0] = entities_(index, nsimd, n + 0, m + 0, l + 0);
            entities[1] = entities_(index, nsimd, n + 0, m + 0, l + 1);
            entities[2] = entities_(index, nsimd, n + 0, m + 1, l + 1);
            entities[3] = entities_(index, nsimd, n + 0, m + 1, l + 0);
            entities[4] = entities_(index, nsimd, n + 1, m + 0, l + 0);
            entities[5] = entities_(index, nsimd, n + 1, m + 0, l + 1);
            entities[6] = entities_(index, nsimd, n + 1, m + 1, l + 1);
            entities[7] = entities_(index, nsimd, n + 1, m + 1, l + 0);
            for (int j = 0; j < 8; ++j) {
              for (int i = 0; i < 8; ++i) {
                lhs[8 * j + i] = stk::simd::get_data(all_lhs(n, m, l, perm[j], perm[i]), nsimd);
              }
            }
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
          }
        }
      }
    }
  }
}


} // namespace nalu
} // namespace Sierra

#endif
