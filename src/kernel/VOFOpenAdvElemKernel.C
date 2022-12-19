// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/VOFOpenAdvElemKernel.h"

#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "BuildTemplates.h"

// template and scratch space
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra::nalu {

template <typename AlgTraits>
VOFOpenAdvElemKernel<AlgTraits>::VOFOpenAdvElemKernel(
  const stk::mesh::MetaData& meta,
  const SolutionOptions& solnOpts,
  const ScalarFieldType& qField,
  const ScalarFieldType& qSpecField,
  const VectorFieldType& dqdxField,
  ElemDataRequests& prereq)
  : qHandle_(qField.mesh_meta_data_ordinal()),
    qbcHandle_(qSpecField.mesh_meta_data_ordinal()),
    dqdxHandle_(dqdxField.mesh_meta_data_ordinal()),
    coordHandle_(get_field_ordinal(meta, solnOpts.get_coordinates_name())),
    flowRateHandle_(
      get_field_ordinal(meta, names::vof::open_flow_rate, meta.side_rank())),
    alphaUpw_(solnOpts.get_alpha_upw_factor(qField.name())),
    hoUpwind_(solnOpts.get_upw_factor(qField.name())),
    skew_(solnOpts.get_skew_symmetric(qField.name()))
{
  static_assert(AlgTraits::nodesPerElement_ == AlgTraits::numFaceIp_);
  prereq.add_cvfem_face_me(
    MasterElementRepo::get_surface_master_element<AlgTraits>());
  prereq.add_coordinates_field(
    coordHandle_, AlgTraits::nDim_, CURRENT_COORDINATES);
  prereq.add_gathered_nodal_field(qHandle_, 1);
  prereq.add_gathered_nodal_field(qbcHandle_, 1);
  prereq.add_gathered_nodal_field(dqdxHandle_, AlgTraits::nDim_);
  prereq.add_face_field(flowRateHandle_, AlgTraits::numFaceIp_);
  prereq.add_gathered_nodal_field(coordHandle_, AlgTraits::nDim_);
  prereq.add_master_element_call(
    skew_ ? FC_SHIFTED_SHAPE_FCN : FC_SHAPE_FCN, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
VOFOpenAdvElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scr)
{
  constexpr int dim = AlgTraits::nDim_;
  constexpr int npe = AlgTraits::nodesPerElement_;

  const auto& q = scr.get_scratch_view_1D(qHandle_);
  const auto& qbc = scr.get_scratch_view_1D(qbcHandle_);
  const auto& vdot = scr.get_scratch_view_1D(flowRateHandle_);
  const auto& dqdx = scr.get_scratch_view_2D(dqdxHandle_);
  const auto& x = scr.get_scratch_view_2D(coordHandle_);

  const auto& me_scr = scr.get_me_views(CURRENT_COORDINATES);
  const auto& interp_mat =
    skew_ ? me_scr.fc_shifted_shape_fcn : me_scr.fc_shape_fcn;

  for (int ip = 0; ip < npe; ++ip) {
    const int nn = ip;

    DoubleType x_interp[3] = {0, 0, 0};
    DoubleType q_interp = 0.0;
    DoubleType qbc_interp = 0.0;
    for (int n = 0; n < npe; ++n) {
      const auto r = interp_mat(ip, n);
      q_interp += r * q(n);
      qbc_interp += r * qbc(n);
      for (int d = 0; d < dim; ++d) {
        x_interp[d] += r * x(n, d);
      }
    }

    DoubleType dq = 0.;
    for (int d = 0; d < dim; ++d) {
      const auto dx = x_interp[d] - x(nn, d);
      dq += dx * dqdx(nn, d);
    }
    const auto q_extrap = q(nn) + stk::math::if_then_else_zero(hoUpwind_, dq);
    const auto q_upw = alphaUpw_ * q_extrap + (1 - alphaUpw_) * q_interp;

    const auto tvdot = vdot(ip);
    const auto in_mask = stk::math::if_then_else_zero(tvdot > 0., 1.);
    rhs(nn) -= tvdot * (in_mask * q_upw + (1 - in_mask) * qbc_interp);
    lhs(nn, nn) += alphaUpw_ * tvdot * in_mask;
    const auto fac = (1 - alphaUpw_) * tvdot * in_mask;
    for (int n = 0; n < npe; ++n) {
      lhs(nn, n) += interp_mat(ip, n) * fac;
    }
  }
}

INSTANTIATE_KERNEL_FACE(VOFOpenAdvElemKernel)

} // namespace sierra::nalu
