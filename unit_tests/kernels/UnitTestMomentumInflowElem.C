/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "kernel/MomentumAdvDiffElemKernel.h"

#ifndef KOKKOS_HAVE_CUDA



namespace {



}

TEST(MomentumInflow, jacobian_is_consistent_with_linearization)
{


}



TEST_F(MomentumKernelHex8Mesh, advection_diffusion)
{
  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.includeDivU_ = 0.0;

  unit_test_utils::HelperObjects helperObjs(bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // Initialize the kernel
  std::unique_ptr<sierra::nalu::Kernel> kernel(
    new sierra::nalu::MomentumAdvDiffElemKernel<sierra::nalu::AlgTraitsHex8>(
      bulk_, solnOpts_, velocity_, viscosity_,
      helperObjs.assembleElemSolverAlg->dataNeededByKernels_));

  // Add to kernels to be tested
  helperObjs.assembleElemSolverAlg->activeKernels_.push_back(kernel.get());

  // Populate LHS and RHS
  helperObjs.assembleElemSolverAlg->execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  namespace gold_values = ::hex8_golds::advection_diffusion;
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->rhs_, gold_values::rhs);
  unit_test_kernel_utils::expect_all_near(helperObjs.linsys->lhs_, gold_values::lhs);
}

#endif

