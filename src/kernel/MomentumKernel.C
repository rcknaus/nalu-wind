/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumKernel.h"

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include "MatrixFreeTraits.h"

#include <master_element/Hex8GeometryFunctions.h>
#include <kernel/TensorProductCVFEMMomentumBDF2TimeDerivative.h>
#include <kernel/TensorProductCVFEMDiffusion.h>
#include <kernel/TensorProductCVFEMMomentumAdvDiff.h>
#include <kernel/TensorProductCVFEMScalarAdvDiff.h>
#include <kernel/TensorProductCVFEMScalarBDF2TimeDerivative.h>

namespace sierra { namespace nalu {

template <int p> nodal_vector_array<DoubleType, p>
momentum_element_residual<p>::residual(
  int index,
  Kokkos::Array<double, 3> gamma,
  const ko::scs_scalar_view<p>& mdot,
  const ko::scalar_view<p>& volume,
  const ko::vector_view<p>& coords,
  const ko::scalar_view<p>& visc,
  const ko::scalar_view<p>& rhom1,
  const ko::scalar_view<p>& rhop0,
  const ko::scalar_view<p>& rhop1,
  const ko::vector_view<p>& velm1,
  const ko::vector_view<p>& velp0,
  const ko::vector_view<p>& velp1,
  const ko::vector_view<p>& Gp)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_vector_array<DoubleType, p>>();
  auto v_rhs = la::make_view(rhs);

  // just pass index in?
  auto elem_velp1 = nodal_vector_view<p,DoubleType>(&velp1(index,0,0,0,0));
  {
    auto elem_coords = nodal_vector_view<p,DoubleType>(&coords(index,0,0,0,0));
    auto elem_visc = nodal_scalar_view<p,DoubleType>(&visc(index,0,0,0));
    auto elem_mdot = scs_scalar_view<p,DoubleType>(&mdot(index,0,0,0,0));
    scs_vector_workview<p, ftype> work_tau_dot_a;
    auto& tau_dot_a = work_tau_dot_a.view();
    tensor_assembly::area_weighted_face_normal_shear_stress(ops, elem_coords, elem_visc, elem_velp1, tau_dot_a);
    tensor_assembly::momentum_advdiff_rhs(ops, tau_dot_a, elem_mdot, elem_velp1, v_rhs);
  }

  auto elem_vol = nodal_scalar_view<p,DoubleType>(&volume(index,0,0,0));
  auto elem_velm1 = nodal_vector_view<p,DoubleType>(&velm1(index,0,0,0,0));
  auto elem_velp0 = nodal_vector_view<p,DoubleType>(&velp0(index,0,0,0,0));
  auto elem_rhom1 = nodal_scalar_view<p,DoubleType>(&rhom1(index,0,0,0));
  auto elem_rhop0 = nodal_scalar_view<p,DoubleType>(&rhop0(index,0,0,0));
  auto elem_rhop1 = nodal_scalar_view<p,DoubleType>(&rhop1(index,0,0,0));
  auto elem_Gp = nodal_vector_view<p,DoubleType>(&Gp(index,0,0,0,0));
  tensor_assembly::momentum_dt_rhs(
    ops, elem_vol, gamma.data(), elem_Gp, elem_rhom1, elem_rhop0, elem_rhop1,
    elem_velm1, elem_velp0, elem_velp1,
    v_rhs
  );
  return rhs;
}

template <int p> nodal_vector_array<DoubleType, p>
momentum_element_residual<p>::linearized_residual(
  int index,
  double gamma,
  const ko::scalar_view<p>& volume,
  const ko::scs_vector_view<p>& area,
  const ko::scs_scalar_view<p>& mdot,
  const nodal_vector_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_vector_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);
  const auto mapped_area = scs_vector_view<p, ftype>(&area(index,0,0,0,0,0));
  const auto elem_mdot = scs_scalar_view<p, ftype>(&mdot(index,0,0,0,0));
  tensor_assembly::split_momentum_advdiff_rhs(ops, mapped_area, elem_mdot, delta, v_rhs);
//
  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  tensor_assembly::momentum_dt_rhs_linearized(ops, vol, gamma, delta, v_rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> momentum_element_residual<p>::linearized_residual(
  int index,
  double gamma,
  const ko::scalar_view<p>& volume,
  const ko::scs_vector_view<p>& area,
  const ko::scs_scalar_view<p>& mdot,
  const nodal_scalar_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);
  const auto mapped_area = scs_vector_view<p, ftype>(&area(index,0,0,0,0,0));
  const auto elem_mdot = scs_scalar_view<p, ftype>(&mdot(index,0,0,0,0));
  tensor_assembly::split_momentum_advdiff_rhs(ops, mapped_area, elem_mdot, delta, v_rhs);
//
  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  tensor_assembly::momentum_dt_rhs_linearized(ops, vol, gamma, delta, v_rhs);
  return rhs;
}


template <int p> nodal_scalar_array<DoubleType, p> momentum_element_residual<p>::lhs_diagonal(
  int index,
  double gamma,
  const ko::scalar_view<p>& volume,
  const ko::scs_vector_view<p>& mapped_area,
  const ko::scs_scalar_view<p>& mdot)
{
  static const auto ops = CVFEMOperators<p>();

  auto lhs = la::zero<matrix_array<ftype, p>>();
  auto v_lhs = la::make_view(lhs);
  const auto elem_mapped_area = scs_vector_view<p, ftype>(&mapped_area(index,0,0,0,0,0));
  const auto elem_mdot = scs_scalar_view<p, ftype>(&mdot(index,0,0,0,0));

  tensor_assembly::scalar_advdiff_lhs(ops, elem_mdot, elem_mapped_area, v_lhs, true);
  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  tensor_assembly::scalar_dt_lhs_diagonal(ops, vol, gamma,  v_lhs);


  constexpr int n1D = p + 1;
  auto diag = nodal_scalar_array<ftype, p>();
  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const int index = idx<n1D>(k, j, i);
        diag(k,j,i) = lhs(index, index);
      }
    }
  }
  return diag;
}
template class momentum_element_residual<POLY1>;
template class momentum_element_residual<POLY2>;
template class momentum_element_residual<POLY3>;
template class momentum_element_residual<POLY4>;

} // namespace nalu
} // namespace Sierra

