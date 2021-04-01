// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

//
#include "ngp_algorithms/SmoothWallFluxesAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

namespace {

KOKKOS_FUNCTION double
compute_utau(
  double up,
  double yp,
  double nu,
  double utau_guess,
  int max_iter = 10,
  double tol = 1e-6)
{
  constexpr double kappa = 0.41;
  constexpr double elog = 9.8;
  const double A = elog * yp / nu;
  double utau = utau_guess;
  for (int k = 0; k < max_iter; ++k) {
    const double wrk = std::log(A * utau);
    const double fPrime = -(1.0 + wrk);
    const double f = kappa * up - utau * wrk;
    const double df = f / fPrime;
    utau -= df;
    if (std::abs(df) < tol) {
      return utau;
    }
  }
  return utau;
}

KOKKOS_FUNCTION DoubleType
compute_utau(
  DoubleType up,
  DoubleType yp,
  DoubleType nu,
  DoubleType utau_guess,
  int len,
  int max_iter = 20,
  double tol = 1e-6)
{
  DoubleType utau(0);
  for (int n = 0; n < len; ++n) {
    const auto up_n = stk::simd::get_data(up, n);
    const auto yp_n = stk::simd::get_data(yp, n);
    const auto nu_n = stk::simd::get_data(nu, n);
    const auto utau_n = stk::simd::get_data(utau_guess, n);
    stk::simd::set_data(
      utau, n, compute_utau(up_n, yp_n, nu_n, utau_n, max_iter, tol));
  }
  return utau;
}

} // namespace

template <typename BcAlgTraits>
SmoothWallFluxesAlg<BcAlgTraits>::SmoothWallFluxesAlg(
  Realm& realm, stk::mesh::Part* part, WallFricVelAlgDriver& algDriver)
  : Algorithm(realm, part),
    algDriver_(algDriver),
    faceData_(realm.meta_data()),
    elemData_(realm.meta_data()),
    velocityNp1_(
      get_field_ordinal(realm.meta_data(), "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(realm.meta_data(), "wall_velocity_bc")),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(realm.meta_data(), "viscosity")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallShearStress_(get_field_ordinal(
      realm.meta_data(),
      "wall_shear_stress_bip",
      realm.meta_data().side_rank())),
    wallNormDist_(get_field_ordinal(
      realm.meta_data(),
      "wall_normal_distance_bip",
      realm.meta_data().side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  elemData_.add_cvfem_surface_me(meSCS_);
  elemData_.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(density_, 1);
  elemData_.add_gathered_nodal_field(viscosity_, 1);

  faceData_.add_cvfem_face_me(meFC_);
  faceData_.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
void
SmoothWallFluxesAlg<BcAlgTraits>::execute()
{
  using FaceElemSimdData =
    sierra::nalu::nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const unsigned velID = velocityNp1_;
  const unsigned bcVelID = bcVelocity_;
  const unsigned rhoID = density_;
  const unsigned muID = viscosity_;
  const unsigned areaVecID = exposedAreaVec_;
  const unsigned wDistID = wallNormDist_;
  auto* meSCS = meSCS_;

  const double kappa = kappa_;
  const double elog = elog_;
  const double yplusCrit = yplusCrit_;

  const stk::mesh::Selector sel =
    realm_.meta_data().locally_owned_part() & stk::mesh::selectUnion(partVec_);

  auto ngpUtau = fieldMgr.template get_field<double>(wallFricVel_);
  const auto utauOps = nalu_ngp::simd_face_elem_field_updater(ngpMesh, ngpUtau);

  auto ngptauSurf = fieldMgr.template get_field<double>(wallShearStress_);
  const auto tauSurfOps =
    nalu_ngp::simd_face_elem_field_updater(ngpMesh, ngptauSurf);

  nalu_ngp::ArraySimdDouble2 utauSum(0.0);
  Kokkos::Sum<nalu_ngp::ArraySimdDouble2> utauReducer(utauSum);

  const std::string algName = "SmoothWallFluxesAlg_" +
                              std::to_string(BcAlgTraits::faceTopo_) + "_" +
                              std::to_string(BcAlgTraits::elemTopo_);

  nalu_ngp::run_face_elem_par_reduce(
    algName, meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(
      FaceElemSimdData & feData, nalu_ngp::ArraySimdDouble2 & uSum) {
      constexpr int dim = BcAlgTraits::nDim_;

      auto& scrViewsElem = feData.simdElemView;
      const auto& vel = scrViewsElem.get_scratch_view_2D(velID);
      const auto& rho = scrViewsElem.get_scratch_view_1D(rhoID);
      const auto& mu = scrViewsElem.get_scratch_view_1D(muID);

      auto& scrViewsFace = feData.simdFaceView;
      const auto& bcvel = scrViewsFace.get_scratch_view_2D(bcVelID);
      const auto& areav = scrViewsFace.get_scratch_view_2D(areaVecID);
      const auto& dtw = scrViewsFace.get_scratch_view_1D(wDistID);

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        const int nodeL = meSCS->opposingNodes(feData.faceOrd, ip);

        DoubleType aMag(0);
        for (int d = 0; d < dim; ++d) {
          aMag += areav(ip, d) * areav(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        NALU_ALIGNED Kokkos::Array<DoubleType, dim> nx;
        for (int d = 0; d < dim; ++d) {
          nx[d] = areav(ip, d) / aMag;
        }

        NALU_ALIGNED Kokkos::Array<DoubleType, dim> uTan;
        for (int i = 0; i < dim; ++i) {
          uTan[i] = 0;
          for (int j = 0; j < dim; ++j) {
            const auto proj = (i == j) - nx[i] * nx[j];
            uTan[i] += proj * (vel(nodeL, j) - bcvel(ip, j));
          }
        }

        DoubleType uTanMag(0);
        for (int i = 0; i < dim; ++i) {
          uTanMag += uTan[i] * uTan[i];
        }
        uTanMag = stk::math::sqrt(uTanMag);

        const auto nu = mu(nodeL) / rho(nodeL);
        const auto utauGuess = yplusCrit * nu / dtw(ip);
        const auto utau =
          compute_utau(uTanMag, dtw(ip), nu, utauGuess, feData.numSimdElems);

        const auto yp = (utau / nu) * dtw(ip);
        const auto lambda = -stk::math::if_then_else(
          yp < yplusCrit, mu(nodeL) / yp,
          rho(nodeL) * utau * kappa / stk::math::log(elog * yp));

        for (int d = 0; d < dim; ++d) {
          tauSurfOps(feData, ip * dim + d) = lambda * uTan[d];
        }
        uSum.array_[0] += utau * aMag;
        uSum.array_[1] += aMag;
      }
    },
    utauReducer);

  algDriver_.accumulate_utau_area_sum(utauSum.array_[0], utauSum.array_[1]);
}
INSTANTIATE_KERNEL_FACE_ELEMENT(SmoothWallFluxesAlg)
} // namespace nalu
} // namespace sierra
