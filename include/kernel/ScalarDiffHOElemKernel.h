/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarDiffHOElemKernel_h
#define ScalarDiffHOElemKernel_h

#include <kernel/Kernel.h>
#include <KokkosInterface.h>
#include <AlgTraits.h>

#include <master_element/TensorProductCVFEMOperators.h>
#include <CVFEMTypeDefs.h>

#include <FieldTypeDef.h>
#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class ElemDataRequests;
class Realm;
class MasterElement;

template<typename AlgTraits>
class ScalarDiffHOElemKernel final : public Kernel
{
  DeclareCVFEMTypeDefs(CVFEMViews<AlgTraits::polyOrder_>);
public:
  ScalarDiffHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests& dataPreReqs);

  ~ScalarDiffHOElemKernel() = default;

  const ScalarFieldType& solution_field() { return *scalarQ_; }

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;


  void mfexecute(nodal_scalar_view& rhs, ScratchViewsHO<DoubleType>& scratchViews);


private:
  ScalarFieldType *scalarQ_{nullptr};
  ScalarFieldType *diffFluxCoeff_{nullptr};
  VectorFieldType *coordinates_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};

  double timer_diff{0};
  double timer_jac{0};
  double timer_resid{0};

};



} // namespace nalu
} // namespace Sierra

#endif
