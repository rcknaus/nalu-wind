// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewe Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/MomentumWallFuncEdgeKernel.h"
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

template <typename BcAlgTraits>
MomentumWallFuncEdgeKernel<BcAlgTraits>::MomentumWallFuncEdgeKernel(
  stk::mesh::MetaData& meta,
  bool slip,
  ElemDataRequests& faceData,
  ElemDataRequests& elemData)
  : NGPKernel<MomentumWallFuncEdgeKernel<BcAlgTraits>>(),
    slip_(slip),
    velocityNp1_(get_field_ordinal(meta, "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(meta, "wall_velocity_bc")),
    density_(get_field_ordinal(meta, "density")),
    viscosity_(get_field_ordinal(meta, "viscosity")),
    exposedAreaVec_(
      get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    wallFricVel_(
      get_field_ordinal(meta, "wall_friction_velocity_bip", meta.side_rank())),
    wallNormDist_(
      get_field_ordinal(meta, "wall_normal_distance_bip", meta.side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceData.add_cvfem_face_me(meFC_);
  elemData.add_cvfem_surface_me(meSCS_);

  constexpr int dim = BcAlgTraits::nDim_;

  faceData.add_gathered_nodal_field(velocityNp1_, dim);
  faceData.add_gathered_nodal_field(bcVelocity_, dim);
  faceData.add_gathered_nodal_field(density_, 1);
  faceData.add_gathered_nodal_field(viscosity_, 1);

  faceData.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, dim);
  faceData.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);
  faceData.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
void
MomentumWallFuncEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& faceViews,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
  int elemFaceOrdinal)
{
  constexpr int dim = BcAlgTraits::nDim_;

  const auto& u = faceViews.get_scratch_view_2D(velocityNp1_);
  const auto& ubc = faceViews.get_scratch_view_2D(bcVelocity_);
  const auto& rho = faceViews.get_scratch_view_1D(density_);
  const auto& mu = faceViews.get_scratch_view_1D(viscosity_);
  const auto& dtw = faceViews.get_scratch_view_1D(wallNormDist_);
  const auto& av = faceViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& utau = faceViews.get_scratch_view_1D(wallFricVel_);

  NGP_ThrowAssert(BcAlgTraits::nodesPerFace_ == BcAlgTraits::numFaceIp_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nn = (slip_) ? meSCS_->ipNodeMap(elemFaceOrdinal)[ip]
                          : meSCS_->opposingNodes(elemFaceOrdinal, ip);
    const int nf = meFC_->ipNodeMap()[ip];

    DoubleType amag = 0.0;
    for (int d = 0; d < dim; ++d) {
      amag += av(ip, d) * av(ip, d);
    }
    amag = stk::math::sqrt(amag);

    NALU_ALIGNED Kokkos::Array<DoubleType, dim> nx;
    for (int d = 0; d < dim; ++d) {
      nx[d] = av(ip, d) / amag;
    }

    const auto yp = (rho(nf) * dtw(nf) / mu(nf)) * utau(ip);
    const auto lambda = stk::math::if_then_else(
      yp < yplusCrit_, mu(nf) / yp,
      rho(nf) * kappa_ * utau(ip) / stk::math::log(elog_ * yp));

    for (int i = 0; i < dim; ++i) {
      const int rowR = nn * dim + i;
      DoubleType uiTan = 0.0;
      DoubleType uiBcTan = 0.0;
      for (int j = 0; j < dim; ++j) {
        const int colR = nn * dim + j;
        const auto proj = (i == j) - nx[i] * nx[j];
        uiTan += proj * u(nn, j);
        uiBcTan += proj * ubc(nn, j);
        lhs(rowR, colR) += lambda * proj * amag;
      }
      rhs(rowR) -= lambda * (uiTan - uiBcTan) * amag;
    }
  }
}
INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumWallFuncEdgeKernel)

} // namespace nalu
} // namespace sierra
