// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "edge_kernels/MomentumABLWallFuncEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"
#include "wind_energy/MoninObukhov.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
MomentumABLWallFuncEdgeKernel<BcAlgTraits>::MomentumABLWallFuncEdgeKernel(
  bool slip,
  stk::mesh::MetaData& meta,
  std::string coordname,
  const double& gravity,
  const double& z0,
  const double& Tref,
  const double& kappa,
  ElemDataRequests& faceDataPreReqs,
  ElemDataRequests& elemData
) : NGPKernel<MomentumABLWallFuncEdgeKernel<BcAlgTraits>>(),
    slip_(slip),
    coordinates_(get_field_ordinal(meta, coordname)),
    velocityNp1_(get_field_ordinal(meta, "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(meta, "wall_velocity_bc")),
    density_(get_field_ordinal(meta, "density")),
    viscosity_(get_field_ordinal(meta, "effective_viscosity_u")), 
    bcHeatFlux_(get_field_ordinal(meta, "heat_flux_bc")),
    specificHeat_(get_field_ordinal(meta, "specific_heat")),
    exposedAreaVec_(get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    wallFricVel_(get_field_ordinal(meta, "wall_friction_velocity_bip", meta.side_rank())),
    wallNormDist_(get_field_ordinal(meta, "wall_normal_distance_bip", meta.side_rank())),
    gravity_(gravity),
    z0_(z0),
    Tref_(Tref),
    kappa_(kappa),
    meFC_(MasterElementRepo::get_surface_master_element<typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<typename BcAlgTraits::ElemTraits>())
{
  faceDataPreReqs.add_cvfem_face_me(meFC_);
  elemData.add_cvfem_surface_me(meSCS_);

  faceDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(density_, 1);
  faceDataPreReqs.add_gathered_nodal_field(viscosity_, 1);

  faceDataPreReqs.add_gathered_nodal_field(bcHeatFlux_, 1);
  faceDataPreReqs.add_gathered_nodal_field(specificHeat_, 1);
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);
  faceDataPreReqs.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);

  elemData.add_coordinates_field(coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
}

template<typename BcAlgTraits>
void
MomentumABLWallFuncEdgeKernel<BcAlgTraits>::no_slip(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  namespace mo = abl_monin_obukhov;

  const DoubleType eps = 1.0e-8;
  const DoubleType Lmax = 1.0e8;

  // Unit normal vector
  NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];

  const auto& v_vel = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_bcvel = scratchViews.get_scratch_view_2D(bcVelocity_);
  const auto& v_density = scratchViews.get_scratch_view_1D(density_);
  const auto& v_bcHeatFlux = scratchViews.get_scratch_view_1D(bcHeatFlux_);
  const auto& v_specificHeat = scratchViews.get_scratch_view_1D(specificHeat_);
  const auto& v_areavec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& v_wallfricvel = scratchViews.get_scratch_view_1D(wallFricVel_);
  const auto& v_wallnormdist = scratchViews.get_scratch_view_1D(wallNormDist_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nodeR = meSCS_->side_node_ordinals(elemFaceOrdinal)[ip];
    const int nodeL = meSCS_->opposingNodes(elemFaceOrdinal, ip);

    DoubleType amag = 0.0;
    DoubleType un = 0.;
    for (int d=0; d < BcAlgTraits::nDim_; ++d) {
      amag += v_areavec(ip, d) * v_areavec(ip, d);
      un += (v_vel(nodeR, d) -  v_bcvel(nodeR, d)) * v_areavec(ip, d);
    }
    amag = stk::math::sqrt(amag);
    un /= amag;

    // unit normal
    for (int d=0; d < BcAlgTraits::nDim_; ++d) {
      nx[d] = v_areavec(ip, d) / amag;
    }

    const DoubleType zh = v_wallnormdist(ip);
    const DoubleType ustar = v_wallfricvel(ip);

    const DoubleType heatflux =v_bcHeatFlux(nodeR);
    const DoubleType rho = v_density(nodeR);
    const DoubleType Cp = v_specificHeat(nodeR);
    const DoubleType Tflux = heatflux / (rho * Cp);

    const DoubleType Lfac = stk::math::if_then_else(
      (stk::math::abs(Tflux) < eps), Lmax, (-Tref_ / (kappa_ * gravity_ * Tflux)));

    DoubleType moLen = ustar * ustar * ustar * Lfac;
    const DoubleType sign = stk::math::if_then_else((Tflux < 0.0), 1.0, -1.0);
    moLen = sign * stk::math::max(1.0e-10, stk::math::abs(moLen));

    const DoubleType zeta = (zh / moLen);
    const DoubleType psim = stk::math::if_then_else(
      (Tflux < -eps), mo::psim_stable(zeta, beta_m_),     // Stable stratification
      stk::math::if_then_else(
        (Tflux > eps), mo::psim_unstable(zeta, gamma_m_), // Unstable stratification
        0.0));                                            // Neutral conditions

    const DoubleType lambda =
      (rho * kappa_ * ustar / (stk::math::log(zh / z0_) - psim)) * amag;

    for (int i=0; i < BcAlgTraits::nDim_; ++i) {
      const int rowL = nodeL * BcAlgTraits::nDim_ + i;
      for (int j=0; j < BcAlgTraits::nDim_; ++j) {
        const int colL = nodeL * BcAlgTraits::nDim_ + j;
        const DoubleType proj = (i==j) ? 1 - nx[i] * nx[j]: -nx[i]*nx[j];
        lhs(rowL, colL) += lambda * proj;
      }
      rhs(rowL) -= lambda * ( (v_vel(nodeL, i) - v_bcvel(nodeL, i)) - un * nx[i]);
    }
  }
}


template<typename BcAlgTraits>
void
MomentumABLWallFuncEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& elemScratchViews,
  int elemFaceOrdinal)
{
  if (!slip_) {
    return this->no_slip(lhs, rhs, scratchViews, elemScratchViews, elemFaceOrdinal);
  }

  namespace mo = abl_monin_obukhov;

  const DoubleType eps = 1.0e-8;
  const DoubleType Lmax = 1.0e8;

  // Unit normal vector
  NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];

  const auto& v_vel = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_bcvel = scratchViews.get_scratch_view_2D(bcVelocity_);
  const auto& v_density = scratchViews.get_scratch_view_1D(density_);
  const auto& v_bcHeatFlux = scratchViews.get_scratch_view_1D(bcHeatFlux_);
  const auto& v_specificHeat = scratchViews.get_scratch_view_1D(specificHeat_);
  const auto& v_areavec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& v_wallfricvel = scratchViews.get_scratch_view_1D(wallFricVel_);
  const auto& v_wallnormdist = scratchViews.get_scratch_view_1D(wallNormDist_);
  const auto& v_visc = scratchViews.get_scratch_view_1D(viscosity_);

  const auto& v_coord = elemScratchViews.get_scratch_view_2D(coordinates_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nodeR = meSCS_->side_node_ordinals(elemFaceOrdinal)[ip];
    const int nodeL = meSCS_->opposingNodes(elemFaceOrdinal, ip);

    DoubleType amag = 0.0;
    DoubleType a_dot_e = 0.0;
    DoubleType un = 0.;
    for (int d=0; d < BcAlgTraits::nDim_; ++d) {
      amag += v_areavec(ip, d) * v_areavec(ip, d);
      const auto evec_d = v_coord(nodeR, d) - v_coord(nodeL, d);
      a_dot_e += evec_d * v_areavec(ip, d);
      un += (v_vel(nodeR, d) -  v_bcvel(nodeR, d)) * v_areavec(ip, d);
    }
    amag = stk::math::sqrt(amag);
    un /= amag;

    // unit normal
    for (int d=0; d < BcAlgTraits::nDim_; ++d) {
      nx[d] = v_areavec(ip, d) / amag;
    }

    const DoubleType zh = v_wallnormdist(ip);
    const DoubleType ustar = v_wallfricvel(ip);

    const DoubleType heatflux =v_bcHeatFlux(nodeR);
    const DoubleType rho = v_density(nodeR);
    const DoubleType Cp = v_specificHeat(nodeR);
    const DoubleType Tflux = heatflux / (rho * Cp);

    const DoubleType Lfac = stk::math::if_then_else(
      (stk::math::abs(Tflux) < eps), Lmax, (-Tref_ / (kappa_ * gravity_ * Tflux)));

    DoubleType moLen = ustar * ustar * ustar * Lfac;
    const DoubleType sign = stk::math::if_then_else((Tflux < 0.0), 1.0, -1.0);
    moLen = sign * stk::math::max(1.0e-10, stk::math::abs(moLen));

    const DoubleType zeta = (zh / moLen);
    const DoubleType psim = stk::math::if_then_else(
      (Tflux < -eps), mo::psim_stable(zeta, beta_m_),     // Stable stratification
      stk::math::if_then_else(
        (Tflux > eps), mo::psim_unstable(zeta, gamma_m_), // Unstable stratification
        0.0));                                            // Neutral conditions

    const DoubleType lambda =
      (rho * kappa_ * ustar / (stk::math::log(zh / z0_) - psim)) * amag;
    
    constexpr double penalty = 1e5;
    const auto sigma = penalty * v_visc(nodeR) * amag * amag / a_dot_e;

    NALU_ALIGNED Kokkos::Array<DoubleType, 3> wall_projected_velocity;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      wall_projected_velocity[d] = (v_vel(nodeR, d) - v_bcvel(nodeR,d)) - un * nx[d];
    }

    for (int i=0; i < BcAlgTraits::nDim_; ++i) {
      const int rowR = nodeR * BcAlgTraits::nDim_ + i;
      for (int j=0; j < BcAlgTraits::nDim_; ++j) {
        const int colR = nodeR * BcAlgTraits::nDim_ + j;
        DoubleType ninj = nx[i] * nx[j];
        const DoubleType kron = (i==j) ? 1 : 0;
        lhs(rowR, colR) += sigma * v_areavec(ip, i) * nx[j] + lambda * (kron-ninj);
      }
      rhs(rowR) -= sigma * un * v_areavec(ip, i) + lambda * wall_projected_velocity[i];
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumABLWallFuncEdgeKernel)

}  // nalu
}  // sierra
