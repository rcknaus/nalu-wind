// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef VOFOpenAdvElemKernel_h
#define VOFOpenAdvElemKernel_h

#include "master_element/MasterElement.h"

// scratch space
#include "ScratchViews.h"

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra::nalu {

namespace names::vof {
inline constexpr auto open_flow_rate = "open_volume_flow_rate"
}

class SolutionOptions;
class ElemDataRequests;

template <typename AlgTraits>
class VOFOpenAdvElemKernel final : public Kernel
{
public:
  using Kernel::execute;

  VOFpenAdvElemKernel(
    const stk::mesh::MetaData& metaData,
    const SolutionOptions& solnOpts,
    const ScalarFieldType& scalarQ,
    const ScalarFieldType& bcScalarQ,
    const VectorFieldType& Gjq,
    ElemDataRequests& elemDataPreReqs);

  void execute(
    SharedMemView<DoubleType**>& lhs,
    SharedMemView<DoubleType*>& rhs,
    ScratchViews<DoubleType>& scr) final;

private:
  const unsigned qHandle_;
  const unsigned qbcHandle_;
  const unsigned dqdxHandle_;
  const unsigned coordHandle_;
  const unsigned flowRateHandle_;
  const double alphaUpw_;
  const double hoUpwind_;
  const double skew_;
};

} // namespace sierra::nalu

#endif
