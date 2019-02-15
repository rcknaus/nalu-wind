/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MomentumCoriolisSrcHOElemKernel_h
#define MomentumCoriolisSrcHOElemKernel_h

#include <kernel/Kernel.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>
#include <CoriolisSrc.h>


#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

// Kokkos
#include <KokkosInterface.h>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class TimeIntegrator;

template<class AlgTraits>
class MomentumCoriolisSrcHOElemKernel final : public Kernel
{
DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  MomentumCoriolisSrcHOElemKernel(
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
  const CoriolisSrc cor_;

  VectorFieldType* coordinates_{nullptr};
  ScalarFieldType* density_{nullptr};
  VectorFieldType* velocity_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_> ops_;


};

} // namespace nalu
} // namespace Sierra

#endif
