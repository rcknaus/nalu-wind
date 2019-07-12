/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorProductCVFEMAdvectionMetric_h
#define TensorProductCVFEMAdvectionMetric_h

#include <master_element/CVFEMCoefficientMatrices.h>
#include <master_element/TensorProductCVFEMOperators.h>
#include <master_element/DirectionMacros.h>
#include <master_element/TensorOps.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <AlgTraits.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {
namespace high_order_metrics {
namespace internal {

//template<int p, typename Scalar>
//void hex_vertex_coordinates(
//  const nodal_vector_view<p, Scalar>& xc,
//  Scalar base_box[3][8])
//{
//  for (int d = 0; d < 3; ++d) {
//    base_box[d][0] = xc(0, 0, 0, d);
//    base_box[d][1] = xc(0, 0, p, d);
//    base_box[d][2] = xc(0, p, p, d);
//    base_box[d][3] = xc(0, p, 0, d);
//
//    base_box[d][4] = xc(p, 0, 0, d);
//    base_box[d][5] = xc(p, 0, p, d);
//    base_box[d][6] = xc(p, p, p, d);
//    base_box[d][7] = xc(p, p, 0, d);
//  }
//}
//
//
//template <int di, int dj, typename Scalar>
//Scalar hex_jacobian_component(
//  const LocalArray<Scalar[3][8]> base_box,
//  double il, double ir,
//  double jl, double jr,
//  double kl, double kr)
//{
//  return (dj == XH) ?
//  ( -jl * kl * base_box(di, 0)
//  +  jl * kl * base_box(di, 1)
//  +  jr * kl * base_box(di, 2)
//  -  jr * kl * base_box(di, 3)
//  -  jl * kr * base_box(di, 4)
//  +  jl * kr * base_box(di, 5)
//  +  jr * kr * base_box(di, 6)
//  -  jr * kr * base_box(di, 7)
//      ) * 0.5
// : (dj == YH) ?
//  ( -il * kl * base_box(di, 0)
//  -  ir * kl * base_box(di, 1)
//  +  ir * kl * base_box(di, 2)
//  +  il * kl * base_box(di, 3)
//  -  il * kr * base_box(di, 4)
//  -  ir * kr * base_box(di, 5)
//  +  ir * kr * base_box(di, 6)
//  +  il * kr * base_box(di, 7)
//  ) * 0.5
//  :
//  ( -il * jl * base_box(di, 0)
//  -  ir * jl * base_box(di, 1)
//  -  ir * jr * base_box(di, 2)
//  -  il * jr * base_box(di, 3)
//  +  il * jl * base_box(di, 4)
//  +  ir * jl * base_box(di, 5)
//  +  ir * jr * base_box(di, 6)
//  +  il * jr * base_box(di, 7)
//   ) * 0.5;
//}
//
//template <int di> LocalArray<DoubleType[3]> area_vector(
//  const LocalArray<DoubleType[3][8]> box,
//  double il, double ir,
//  double jl, double jr,
//  double kl, double kr)
//{
//  constexpr int ds1 = (di == XH) ? ZH : (di == YH) ? XH : YH;
//  constexpr int ds2 = (di == XH) ? YH : (di == YH) ? ZH : XH;
//
//  const auto dx_ds1 = hex_jacobian_component<XH,ZH>(box, il, ir, jl, jr, kl, kr);
//  const auto dx_ds2 = hex_jacobian_component<XH,YH>(box, il, ir, jl, jr, kl, kr);
//
//  const auto dy_ds1 = hex_jacobian_component<YH,ZH>(box, il, ir, jl, jr, kl, kr);
//  const auto dy_ds2 = hex_jacobian_component<YH,YH>(box, il, ir, jl, jr, kl, kr);
//
//  const auto dz_ds1 = hex_jacobian_component<ZH,ZH>(box, il, ir, jl, jr, kl, kr);
//  const auto dz_ds2 = hex_jacobian_component<ZH,YH>(box, il, ir, jl, jr, kl, kr);
//  return {{ dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2, dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2, dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2 }};
//}
//}
//
}

template <int p, typename Scalar>
void compute_area_linear(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  scs_vector_view<p, Scalar>& metric)
{
  const auto& nodalInterp = ops.mat_.linearNodalInterp;
  const auto& scsInterp = ops.mat_.linearScsInterp;

  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };
        NALU_ALIGNED Scalar areav[3]  = {0.,0.,0.};
        hex_areav_x(base_box, interpi, interpj, interpk, areav);

        metric(XH, k, j, i, XH) = areav[XH];
        metric(XH, k, j, i, YH) = areav[YH];
        metric(XH, k, j, i, ZH) = areav[ZH];
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
        NALU_ALIGNED Scalar areav[3] = {0.,0.,0.};
        hex_areav_y(base_box, interpi, interpj, interpk, areav);

        metric(YH, k, j, i, XH) = areav[XH];
        metric(YH, k, j, i, YH) = areav[YH];
        metric(YH, k, j, i, ZH) = areav[ZH];
      }
    }
  }

  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
        NALU_ALIGNED Scalar areav[3] = {0.,0.,0.};
        hex_areav_z(base_box, interpi, interpj, interpk, areav);

        metric(ZH, k, j, i, XH) = areav[XH];
        metric(ZH, k, j, i, YH) = areav[YH];
        metric(ZH, k, j, i, ZH) = areav[ZH];
      }
    }
  }
}

namespace {
template<int p, typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION
void face_vertex_coordinates(
  const face_nodal_vector_view<Scalar, p>& xc,
  Scalar base_box[3][4])
{
  for (int d = 0; d < 3; ++d) {
    base_box[d][0] = xc(0, 0, d);
    base_box[d][1] = xc(0, p, d);
    base_box[d][2] = xc(p, p, d);
    base_box[d][3] = xc(p, 0, d);
  }
}

template <int di, int dj, typename Scalar> Scalar jacobian_component
( const Scalar base_box[3][4], const Scalar interpi[2], const Scalar interpj[2])
{
  return (dj == XH) ?
   ( -interpj[0] * base_box[di][0]
   +  interpj[0] * base_box[di][1]
   +  interpj[1] * base_box[di][2]
   -  interpj[1] * base_box[di][3]
   ) * 0.5
   :
   ( -interpi[0] * base_box[di][0]
   -  interpi[1] * base_box[di][1]
   +  interpi[1] * base_box[di][2]
   +  interpi[0] * base_box[di][3]
   ) * 0.5;
}


template <typename Scalar>
void face_area(const Scalar base_box[3][4], const Scalar interpi[2], const Scalar interpj[2],
  Scalar areav[3])
{
    static constexpr int ds1 = XH;
    static constexpr int ds2 = YH;
    const auto dx_ds1 = jacobian_component<XH, ds1>(base_box, interpi, interpj);
    const auto dx_ds2 = jacobian_component<XH, ds2>(base_box, interpi, interpj);

    const auto dy_ds1 = jacobian_component<YH, ds1>(base_box, interpi, interpj);
    const auto dy_ds2 = jacobian_component<YH, ds2>(base_box, interpi, interpj);

    const auto dz_ds1 = jacobian_component<ZH, ds1>(base_box, interpi, interpj);
    const auto dz_ds2 = jacobian_component<ZH, ds2>(base_box, interpi, interpj);

    areav[0] = dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2;
    areav[1] = dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2;
    areav[2] = dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2;
}


}

template <int p, typename Scalar>
void compute_exposed_area_linear(
  const CVFEMOperators<p, Scalar>& ops,
  const face_nodal_vector_view<Scalar, p>& xc,
  face_nodal_vector_view<Scalar, p>& metric)
{
  const auto& nodalInterp = ops.mat_.linearNodalInterp;

  NALU_ALIGNED Scalar base_box[3][4];
  face_vertex_coordinates<p, Scalar>(xc, base_box);

  for (int j = 0; j < p + 1; ++j) {
    NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
    for (int i = 0; i < p+1; ++i) {
      NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
      NALU_ALIGNED Scalar areav[3]  = {0.,0.,0.};
      face_area(base_box, interpi, interpj, areav);

      metric(j, i, XH) = areav[XH];
      metric(j, i, YH) = areav[YH];
      metric(j, i, ZH) = areav[ZH];
    }
  }
}

template <int p, typename Scalar>
void compute_mdot_linear(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  const scs_vector_view<p, Scalar>& laplacian_metric,
  double projTimeScale,
  const nodal_scalar_view<p, Scalar>& density,
  const nodal_vector_view<p, Scalar>& velocity,
  const nodal_vector_view<p, Scalar>& proj_pressure_gradient,
  const nodal_scalar_view<p, Scalar>& pressure,
  scs_scalar_view<p, Scalar>& mdot)
{
  const auto& nodalInterp = ops.mat_.linearNodalInterp;
  const auto& scsInterp = ops.mat_.linearScsInterp;

  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  nodal_vector_array<Scalar, p> work_rhou_corr;
  auto rhou_corr = la::make_view(work_rhou_corr);

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        const auto rho = density(k, j, i);
        for (int d = 0; d < 3; ++d) {
          rhou_corr(k, j, i, d) = rho * velocity(k, j, i, d) + projTimeScale * proj_pressure_gradient(k, j, i, d);
        }
      }
    }
  }

  nodal_vector_array<Scalar, p> work_rhou_corrIp;
  auto rhou_corrIp = la::make_view(work_rhou_corrIp);

  nodal_vector_array<Scalar, p> work_dpdxhIp;
  auto dpdxhIp = la::make_view(work_dpdxhIp);

  ops.scs_xhat_interp(rhou_corr, rhou_corrIp);
  ops.scs_xhat_grad(pressure, dpdxhIp);

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { scsInterp(0, i), scsInterp(1, i) };
        NALU_ALIGNED Scalar areav[3];
        hex_areav_x(base_box, interpi, interpj, interpk, areav);

        const auto dpdxIp_dot_A = laplacian_metric(XH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                + laplacian_metric(XH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                + laplacian_metric(XH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

        const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                  + rhou_corrIp(k, j, i, YH) * areav[YH]
                                  + rhou_corrIp(k, j, i, ZH) * areav[ZH];

        mdot(XH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
      }
    }
  }

  ops.scs_yhat_interp(rhou_corr, rhou_corrIp);
  ops.scs_yhat_grad(pressure, dpdxhIp);

  for (int k = 0; k < p + 1; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < p; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { scsInterp(0, j), scsInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
        NALU_ALIGNED Scalar areav[3];
        hex_areav_y(base_box, interpi, interpj, interpk, areav);

        const auto dpdxIp_dot_A = laplacian_metric(YH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                + laplacian_metric(YH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                + laplacian_metric(YH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

        const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                  + rhou_corrIp(k, j, i, YH) * areav[YH]
                                  + rhou_corrIp(k, j, i, ZH) * areav[ZH];

        mdot(YH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
      }
    }
  }

  ops.scs_zhat_interp(rhou_corr, rhou_corrIp);
  ops.scs_zhat_grad(pressure, dpdxhIp);
  for (int k = 0; k < p ; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { scsInterp(0, k), scsInterp(1, k) };
    for (int j = 0; j < p + 1; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < p + 1; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };
        NALU_ALIGNED Scalar areav[3];
        hex_areav_z(base_box, interpi, interpj, interpk, areav);

        const auto dpdxIp_dot_A = laplacian_metric(ZH, k, j, i, XH) * dpdxhIp(k, j, i, XH)
                                + laplacian_metric(ZH, k, j, i, YH) * dpdxhIp(k, j, i, YH)
                                + laplacian_metric(ZH, k, j, i, ZH) * dpdxhIp(k, j, i, ZH);

        const auto rhouCorr_dot_A = rhou_corrIp(k, j, i, XH) * areav[XH]
                                  + rhou_corrIp(k, j, i, YH) * areav[YH]
                                  + rhou_corrIp(k, j, i, ZH) * areav[ZH];

        mdot(ZH, k, j, i) = rhouCorr_dot_A - projTimeScale * dpdxIp_dot_A;
      }
    }
  }
}
} // namespace high_order_metrics
} // namespace nalu
} // namespace sierra

#endif
