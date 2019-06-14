/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef CoefficientMatrices_h
#define CoefficientMatrices_h

#include <master_element/CVFEMCoefficients.h>
#include <CVFEMTypeDefs.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu{
//
template <int p, typename Scalar> struct CoefficientMatrices
{
  using value_type = double;
  constexpr static int poly_order = p;
  const scs_matrix_array<double, p> scsDeriv{coeffs::scs_derivative_weights<double, p>()};
  const scs_matrix_array<double, p> scsInterp{coeffs::scs_interpolation_weights<double, p>()};
  const nodal_matrix_array<double, p> nodalWeights{coeffs::nodal_integration_weights<double, p>()};
  const nodal_matrix_array<double, p> lumpedNodalWeights{coeffs::lumped_nodal_integration_weights<double, p>()};
  const nodal_matrix_array<double, p> nodalDeriv{coeffs::nodal_derivative_weights<double, p>()};
  const linear_nodal_matrix_array<double, p> linearNodalInterp{coeffs::linear_nodal_interpolation_weights<double, p>()};
  const linear_scs_matrix_array<double, p> linearScsInterp{coeffs::linear_scs_interpolation_weights<double, p>()};
};

//template <int p, typename Scalar> struct CoefficientMatrices
//{
//  using value_type = double;
//  constexpr static int poly_order = p;
//  const scs_matrix_array<double, p> scsDeriv{coeffs::scs_derivative_weights<double, p>()};
//  const scs_matrix_array<double, p> scsInterp{coeffs::scs_interpolation_weights<double, p>()};
//  const nodal_matrix_array<double, p> nodalWeights{coeffs::lumped_nodal_integration_weights<double, p>()};
//  const nodal_matrix_array<double, p> lumpedNodalWeights{coeffs::lumped_nodal_integration_weights<double, p>()};
//  const nodal_matrix_array<double, p> nodalDeriv{coeffs::nodal_derivative_weights<double, p>()};
//  const linear_nodal_matrix_array<double, p> linearNodalInterp{coeffs::linear_nodal_interpolation_weights<double, p>()};
//  const linear_scs_matrix_array<double, p> linearScsInterp{coeffs::linear_scs_interpolation_weights<double, p>()};
//};


} // namespace naluUnit
} // namespace Sierra

#endif

