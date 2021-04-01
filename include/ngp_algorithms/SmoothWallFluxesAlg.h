// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SMOOTHWALLFLUXESALG_H
#define SMOOTHWALLFLUXESALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "ngp_algorithms/WallFricVelAlgDriver.h"

#include "stk_mesh/base/Types.hpp"

#include "NaluParsing.h"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
class SmoothWallFluxesAlg : public Algorithm
{
public:
  SmoothWallFluxesAlg(Realm&, stk::mesh::Part*, WallFricVelAlgDriver&);
  void execute() final;

private:
  WallFricVelAlgDriver& algDriver_;

  ElemDataRequests faceData_;
  ElemDataRequests elemData_;

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_{stk::mesh::InvalidOrdinal};
  unsigned wallShearStress_{stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_{stk::mesh::InvalidOrdinal};

  const double kappa_{0.41};
  const double yplusCrit_{11.63};
  const double elog_{9.8};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* ABLWALLFLUXESALG_H */
