/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corp.                                           */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarInflowElemKernel_h
#define ScalarInflowElemKernel_h

#include "FieldTypeDef.h"
#include "kernel/Kernel.h"

#include <stk_mesh/base/BulkData.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;
class TimeIntegrator;

template<typename BcAlgTraits>
class ScalarInflowElemKernel: public Kernel
{
  static constexpr int dim = BcAlgTraits::nDim_;
  static constexpr double inviscid_penalty = 0.5; // needs to be > 0.5 for inviscid flow
  static constexpr double viscous_penalty = 2.0; // needs to be > 2.0 for stokes flow
public:
  ScalarInflowElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    ScalarFieldType& q,
    ScalarFieldType& qBC,
    ScalarFieldType& diffusivity,
    ElemDataRequests& faceDataPreReqs,
    ElemDataRequests& elemDataPreReqs);

  using Kernel::execute;

  void execute(
    SharedMemView<DoubleType**>& lhs,
    SharedMemView<DoubleType *>& rhs,
    ScratchViews<DoubleType>& faceViews,
    ScratchViews<DoubleType>& elemViews,
    int elemFaceOrdinal);

private:
  ScalarInflowElemKernel() = delete;

  ScalarFieldType* density_{nullptr};
  VectorFieldType* velocityBC_{nullptr};
  ScalarFieldType* qBC_{nullptr};
  ScalarFieldType* q_{nullptr};
  ScalarFieldType* diff_{nullptr};


  VectorFieldType* coordinates_{nullptr};
  GenericFieldType* areav_{nullptr};

  const bool skewSymmetric_;
  double projTimeScale_;

  const int *ipNodeMap_{nullptr};

  MasterElement* meSCS_;

  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_shape_function_ {"vf_shape_func"};
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_adv_shape_function_ {"vf_adv_shape_function"};
};

}  // nalu
}  // sierra

#endif /* ScalarInflowElemKernel_h */
