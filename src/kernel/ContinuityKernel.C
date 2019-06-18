/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/ContinuityKernel.h>

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include "MatrixFreeTraits.h"

#include <master_element/Hex8GeometryFunctions.h>
#include <kernel/TensorProductCVFEMPressurePoisson.h>
#include <kernel/TensorProductCVFEMDiffusion.h>

namespace sierra { namespace nalu {

template <int p> nodal_scalar_array<DoubleType, p> continuity_element_residual<p>::residual(int index,
    double projTimeScale,
    const ko::scs_scalar_view<p>& mdot,
    const ko::scalar_view<p>& qp1)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<DoubleType, p>>();
  auto v_rhs = la::make_view(rhs);

  // these should be subviews
  auto elem_mdot = scs_scalar_view<p, ftype>(&mdot(index,0,0,0,0));
  auto elem_qp1 = nodal_scalar_view<p, ftype>(&qp1(index,0,0,0));
  tensor_assembly::pressure_poisson_rhs(ops, projTimeScale, elem_mdot, v_rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> continuity_element_residual<p>::linearized_residual(
    int index,
    const ko::scs_vector_view<p>& mapped_area,
    const nodal_scalar_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);
  auto metric = scs_vector_view<p, ftype>(&mapped_area(index,0,0,0,0,0));
  tensor_assembly::scalar_diffusion_rhs(ops, metric, delta, v_rhs);
  return rhs;
}
template class continuity_element_residual<POLY1>;
template class continuity_element_residual<POLY2>;
template class continuity_element_residual<POLY3>;
template class continuity_element_residual<POLY4>;

} // namespace nalu
} // namespace Sierra

