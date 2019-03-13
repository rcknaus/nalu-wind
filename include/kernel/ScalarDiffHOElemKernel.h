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

#include <MatrixFreeElemSolver.h>

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
  static constexpr int poly_order = AlgTraits::polyOrder_;
  static constexpr int ndof = 1;

  ScalarDiffHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests& dataPreReqs);

  ~ScalarDiffHOElemKernel() = default;

  ScalarFieldType& solution_field() { return *scalarQ_; }

  using Kernel::execute;
  void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViewsHO<DoubleType>&) final;

  void executemf(nodal_scalar_view& rhs, ScratchViewsHOMF<poly_order>& scratchViews);


private:
  ScalarFieldType *scalarQ_{nullptr};
  ScalarFieldType *diffFluxCoeff_{nullptr};
  VectorFieldType *coordinates_{nullptr};

  CVFEMOperators<AlgTraits::polyOrder_> ops_{};

  double timer_diff{0};
  double timer_jac{0};
  double timer_resid{0};

};


template <int p>
class ScalarConductionHOElemKernel
{
  DeclareCVFEMTypeDefs(CVFEMViews<p>);
public:
  static constexpr int ndof = 1;

  ScalarConductionHOElemKernel(
    const stk::mesh::BulkData& bulkData,
    SolutionOptions& solnOpts,
    ScalarFieldType&  scalarQ,
    ScalarFieldType&  diffFluxCoeff,
    ElemDataRequests& dataPreReqs);

  ~ScalarConductionHOElemKernel() = default;

  ScalarFieldType& solution_field() { return *q_[stk::mesh::StateNP1]; }
  void setup(const TimeIntegrator& ti);
  void executemf(nodal_scalar_view& rhs, ScratchViewsHOMF<p>& scratchViews);


private:
  Kokkos::Array<ScalarFieldType*, 3> q_{{}};
  Kokkos::Array<ScalarFieldType*, 3> rho_{{}};
  Kokkos::Array<DoubleType, 3> gamma_{{}};

  ScalarFieldType* diffusivity_{nullptr};
  VectorFieldType* coordinates_{nullptr};
  CVFEMOperators<p> ops_{};

};



} // namespace nalu
} // namespace Sierra

#endif
