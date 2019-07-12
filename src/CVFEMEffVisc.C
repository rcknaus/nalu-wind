/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "CVFEMEffVisc.h"

#include "MatrixFreeTraits.h"
#include <CVFEMTypeDefs.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu {

namespace {

template <int p, typename Scalar>
nodal_tensor_array<Scalar, p> nodal_grad_u(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  const nodal_vector_view<p, Scalar>& vel)
{
  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  nodal_tensor_array<Scalar, p> work_gradu;
  auto nodal_gradu = la::make_view(work_gradu);

  nodal_tensor_array<Scalar, p> phys_gradu;

  const auto& nodalInterp = ops.mat_.linearNodalInterp;

  ops.nodal_grad(vel, nodal_gradu);
  static constexpr int n1D = p + 1;
  for (int k = 0; k < n1D; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < n1D; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < n1D; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jact[3][3];
        hex_jacobian_t(base_box, interpi, interpj, interpk, jact);

        NALU_ALIGNED Scalar invJac[3][3];
        invert_matrix33(jact, invJac);

        for (int d = 0; d < 3; ++d)  {
          phys_gradu(k, j, i, d, XH) = invJac[XH][XH] * nodal_gradu(k, j, i, d, XH)
                        + invJac[XH][YH] * nodal_gradu(k, j, i, d, YH) + invJac[XH][ZH] * nodal_gradu(k, j, i, d, ZH);

          phys_gradu(k, j, i, d, YH) = invJac[YH][XH] * nodal_gradu(k, j, i, d, XH)
                        + invJac[YH][YH] * nodal_gradu(k, j, i, d, YH) + invJac[YH][ZH] * nodal_gradu(k, j, i, d, ZH);

          phys_gradu(k, j, i, d, ZH) = invJac[ZH][XH] * nodal_gradu(k, j, i, d, XH)
                        + invJac[ZH][YH] * nodal_gradu(k, j, i, d, YH) + invJac[ZH][ZH] * nodal_gradu(k, j, i, d, ZH);
        }
      }
    }
  }
  return phys_gradu;
}

template <typename Scalar>
Scalar sij_magnitude(const LocalArray<Scalar[3][3]>& dudx)
{
  Scalar sijmag = 0;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i <3; ++i) {
      const auto sij = 0.5*(dudx(j,i)+dudx(i,j));
      sijmag += sij*sij;
    }
  }
  return stk::math::sqrt(2*sijmag);
}

template <typename Scalar>
Scalar wale_turbulent_viscosity(double coeff, Scalar density, Scalar filterScale, const LocalArray<Scalar[3][3]>& dudx)
{
  Scalar SijSq = 0.0;
  Scalar SijdSq = 0.0;
  for (int i = 0; i < 3; ++i)  {
    for (int j = 0; j < 3; ++j) {
      const Scalar Sij = 0.5 * (dudx(i, j) + dudx(j, i));
      Scalar gijSq = 0.0;
      Scalar gjiSq = 0.0;
      for (int l = 0; l < 3; ++l) {
        gijSq += dudx(i, l) * dudx(l, j);
        gjiSq += dudx(j, l) * dudx(l, i);
      }
      const Scalar Sijd = 0.5 * (gijSq + gjiSq);
      SijSq += Sij * Sij;
      SijdSq += Sijd * Sijd;
    }
  }

  constexpr double small = 1.0e-8;
  const auto Ls = coeff * filterScale;
  const auto numer = stk::math::pow(SijdSq, 1.5) + small*small;
  const auto denom = stk::math::pow(SijSq, 2.5) + stk::math::pow(SijdSq, 1.25) + small;
  return (density * Ls * Ls * numer / denom);
}

template <typename Scalar>
Scalar smagorinsky_turbulent_viscosity(double coeff, Scalar density, Scalar filterScale, const LocalArray<Scalar[3][3]>& dudx)
{
  const auto smagorinsky_length = (coeff * filterScale);
  return density * smagorinsky_length * smagorinsky_length * sij_magnitude(dudx);
}

}

template <int p> elem_view::scalar_view<p> nodal_smagorinsky_viscosity(
  double smagCoeff,
  elem_view::vector_view<p> coordinates,
  elem_view::scalar_view<p> density,
  elem_view::scalar_view<p> visc,
  elem_view::scalar_view<p> dnv,
  elem_view::vector_view<p> vel)
{
  auto ops = CVFEMOperators<p, DoubleType>();

  auto nodal_smagorinsky = elem_view::scalar_view<p>("nodal_smagorinsky" + std::to_string(rand()),
    coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto local_vel = nodal_vector_view<p,DoubleType>(&vel(index,0,0,0,0));
    const auto dudx = nodal_grad_u<p>(ops, local_coords, local_vel);

    static constexpr int n1D = p + 1;
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto filterScale = stk::math::cbrt(dnv(index, k, j, i));

          LocalArray<DoubleType[3][3]> local_grad_u;
          for (int dj = 0; dj < 3; ++dj) {
            for (int di = 0; di < 3; ++di) {
              local_grad_u(dj, di) = dudx(k, j, i, dj, di);
            }
          }
          const auto rho = density(index, k, j, i);
          const auto turb_visc = smagorinsky_turbulent_viscosity(smagCoeff, rho, filterScale, local_grad_u);
          nodal_smagorinsky(index, k, j, i) = visc(index, k, j, i) + turb_visc;
        }
      }
    }
  }
  return nodal_smagorinsky;
}

template elem_view::scalar_view<POLY1> nodal_smagorinsky_viscosity<POLY1>(
  double, elem_view::vector_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::vector_view<POLY1>);

template elem_view::scalar_view<POLY2> nodal_smagorinsky_viscosity<POLY2>(
  double, elem_view::vector_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::vector_view<POLY2>);

template elem_view::scalar_view<POLY3> nodal_smagorinsky_viscosity<POLY3>(
  double, elem_view::vector_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::vector_view<POLY3>);

template elem_view::scalar_view<POLY4> nodal_smagorinsky_viscosity<POLY4>(
  double, elem_view::vector_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::vector_view<POLY4>);

template <int p> elem_view::scalar_view<p> nodal_wale_viscosity(
  double smagCoeff,
  elem_view::vector_view<p> coordinates,
  elem_view::scalar_view<p> density,
  elem_view::scalar_view<p> visc,
  elem_view::scalar_view<p> dnv,
  elem_view::vector_view<p> vel)
{
  auto ops = CVFEMOperators<p, DoubleType>();

  auto effective_viscosity = elem_view::scalar_view<p>("nodal_wale" + std::to_string(rand()),
    coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto local_vel = nodal_vector_view<p,DoubleType>(&vel(index,0,0,0,0));
    const auto dudx = nodal_grad_u<p>(ops, local_coords, local_vel);

    static constexpr int n1D = p + 1;
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto filterScale = stk::math::cbrt(dnv(index, k, j, i));

          LocalArray<DoubleType[3][3]> local_grad_u;
          for (int dj = 0; dj < 3; ++dj) {
            for (int di = 0; di < 3; ++di) {
              local_grad_u(dj, di) = dudx(k, j, i, dj, di);
            }
          }

          const auto rho = density(index, k, j, i);
          const auto turb_visc = wale_turbulent_viscosity(smagCoeff, rho, filterScale, local_grad_u);
          effective_viscosity(index, k, j, i) = visc(index, k, j, i) + turb_visc;
        }
      }
    }
  }
  return effective_viscosity;
}

template elem_view::scalar_view<POLY1> nodal_wale_viscosity<POLY1>(
  double, elem_view::vector_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::vector_view<POLY1>);

template elem_view::scalar_view<POLY2> nodal_wale_viscosity<POLY2>(
  double, elem_view::vector_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::vector_view<POLY2>);

template elem_view::scalar_view<POLY3> nodal_wale_viscosity<POLY3>(
  double, elem_view::vector_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::vector_view<POLY3>);

template elem_view::scalar_view<POLY4> nodal_wale_viscosity<POLY4>(
  double, elem_view::vector_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::vector_view<POLY4>);

}}
