// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewe Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMWALLFUNCEDGEKERNEL_H
#define MOMENTUMWALLFUNCEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
class MomentumWallFuncEdgeKernel
  : public NGPKernel<MomentumWallFuncEdgeKernel<BcAlgTraits>>
{
public:
  MomentumWallFuncEdgeKernel(
    stk::mesh::MetaData&, bool, ElemDataRequests&, ElemDataRequests&);

  KOKKOS_DEFAULTED_FUNCTION MomentumWallFuncEdgeKernel() = default;
  KOKKOS_DEFAULTED_FUNCTION virtual ~MomentumWallFuncEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:
  const bool slip_{true};
  const double kappa_{0.41};
  const double yplusCrit_{11.63};
  const double elog_{9.8};

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_{stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_{stk::mesh::InvalidOrdinal};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMWALLFUNCEDGEKERNEL_H */
