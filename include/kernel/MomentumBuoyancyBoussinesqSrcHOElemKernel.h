/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MomentumBoussinesqBuoyancySrcHOElemKernel_h
#define MomentumBoussinesqBuoyancySrcHOElemKernel_h

#include <kernel/Kernel.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

// Kokkos
#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class TimeIntegrator;

template<class AlgTraits>
class MomentumBoussinesqBuoyancySrcHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  MomentumBoussinesqBuoyancySrcHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs);

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

private:
  const double rhoRef_;
  const std::array<double,3> gravity_;
  const double tRef_;
  const double beta_;

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* temperatureField_{nullptr};


  CVFEMOperators<AlgTraits::polyOrder_> ops_;
};

} // namespace nalu
} // namespace Sierra

#endif
