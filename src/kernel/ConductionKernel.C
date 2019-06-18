/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/ConductionKernel.h>

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>
#include <master_element/DirectionMacros.h>

#include "MatrixFreeTraits.h"

#include <master_element/Hex8GeometryFunctions.h>
#include <kernel/TensorProductCVFEMScalarBDF2TimeDerivative.h>
#include <kernel/TensorProductCVFEMDiffusion.h>

namespace sierra { namespace nalu {

template <int p> nodal_scalar_array<DoubleType, p> conduction_element_residual<p>::residual(int index,
    Kokkos::Array<double, 3> gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& mapped_area,
    const ko::scalar_view<p>& qm1,
    const ko::scalar_view<p>& qp0,
    const ko::scalar_view<p>& qp1)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<DoubleType, p>>();
  auto v_rhs = la::make_view(rhs);

  // these should be subviews
  auto metric = scs_vector_view<p, ftype>(&mapped_area(index,0,0,0,0,0));
  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  auto elem_qp0 = nodal_scalar_view<p, ftype>(&qp0(index,0,0,0));
  auto elem_qm1 = nodal_scalar_view<p, ftype>(&qm1(index,0,0,0));

  auto elem_qp1 = nodal_scalar_view<p, ftype>(&qp1(index,0,0,0));
  tensor_assembly::scalar_diffusion_rhs(ops, metric, elem_qp1, v_rhs);
  tensor_assembly::scalar_dt_rhs(ops, vol, gamma.data(), elem_qm1, elem_qp0, elem_qp1, v_rhs);
  return rhs;
}

template <int p> nodal_scalar_array<DoubleType, p> conduction_element_residual<p>::linearized_residual(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& mapped_area,
    const nodal_scalar_view<p, DoubleType>& delta)
{
  static const auto ops = CVFEMOperators<p>();

  auto rhs = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_rhs = la::make_view(rhs);
  auto metric = scs_vector_view<p, ftype>(&mapped_area(index,0,0,0,0,0));
  tensor_assembly::scalar_diffusion_rhs(ops, metric, delta, v_rhs);

  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  tensor_assembly::scalar_dt_rhs_linearized(ops, vol, gamma, delta, v_rhs);
  return rhs;
}


template <int p> nodal_scalar_array<DoubleType, p>
conduction_element_residual<p>::lhs_diagonal(
    int index,
    double gamma,
    const ko::scalar_view<p>& volume,
    const ko::scs_vector_view<p>& mapped_area)
{
  // fixme: only build the diagonal
  static const auto ops = CVFEMOperators<p>();

  auto lhs = la::zero<matrix_array<ftype, p>>();
  auto v_lhs = la::make_view(lhs);
  auto metric = scs_vector_view<p, ftype>(&mapped_area(index,0,0,0,0,0));


  tensor_assembly::scalar_diffusion_lhs(ops, metric, v_lhs);

  auto vol = nodal_scalar_view<p, ftype>(&volume(index,0,0,0));
  auto diag = la::zero<nodal_scalar_array<ftype, p>>();
  auto v_diag = la::make_view(diag);
  tensor_assembly::scalar_dt_lhs_diagonal(ops, vol, gamma,  v_diag);

  constexpr int n1D = p + 1;

  for (int k = 0; k < n1D; ++k) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const int index = idx<n1D>(k, j, i);
        diag(k,j,i) += lhs(index, index);
      }
    }
  }
  return diag;
}
template class conduction_element_residual<POLY1>;
template class conduction_element_residual<POLY2>;
template class conduction_element_residual<POLY3>;
template class conduction_element_residual<POLY4>;

#ifdef SPECIALIZE_CONDUCTION_P1
template <> nodal_scalar_array<DoubleType, 1>
conduction_element_residual<1>::rhs(int index,
    Kokkos::Array<double, 3> gamma,
    const ko::scalar_view<1>& volume,
    const ko::scs_vector_view<1>& mapped_area,
    const ko::scalar_view<1>& qm1,
    const ko::scalar_view<1>& qp0,
    const nodal_scalar_view<1, DoubleType>& elem_qp1)
{
  static constexpr double dv[2] = {-0.5, 0.5};
  auto rhs = la::zero<nodal_scalar_array<DoubleType, 1>>();
  auto v_rhs = la::make_view(rhs);

  auto* KOKKOS_RESTRICT rhs_data = rhs.data();
  const auto* KOKKOS_RESTRICT const qnodal = &elem_qp1(0,0,0);
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      const int rowIndex = 4 * k + 2 * j;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 1;

      const auto* KOKKOS_RESTRICT const aj = &mapped_area(index, XH, k, j, 0, 0);
      const auto flux = 0.5 * (aj[XH] +  dv[j] * aj[YH] + dv[k] * aj[ZH]) * (qnodal[indexR] - qnodal[indexL]);
      rhs_data[indexL] -= flux;
      rhs_data[indexR] += flux;
    }
  }

  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 2; ++i) {
      const int rowIndex = 4 * k + i;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 2;

      const auto* KOKKOS_RESTRICT const aj = &mapped_area(index, YH, k, 0, i, 0);
      const auto flux = 0.5 * (dv[i] * aj[XH] + aj[YH] + dv[k] * aj[ZH]) * (qnodal[indexR] - qnodal[indexL]);
      rhs_data[indexL] -= flux;
      rhs_data[indexR] += flux;
    }
  }

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const int rowIndex = 2 * j + i;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 4;

      const auto* KOKKOS_RESTRICT const aj = &mapped_area(index, ZH, 0, j, i, 0);
      const auto flux =  0.5 * (dv[i] * aj[XH] + dv[j] * aj[YH] + aj[ZH]) * (qnodal[indexR] - qnodal[indexL]);
      rhs_data[indexL] -= flux;
      rhs_data[indexR] += flux;
    }
  }

  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        rhs(k, j, i) -= volume(index, k, j, i)
                     * (gamma[0] * qp1(index, k, j, i) + gamma[1] * qp0(index, k, j, i) + gamma[2] * qm1(index, k, j, i));
      }
    }
  }
  return rhs;
}
#endif


} // namespace nalu
} // namespace Sierra

