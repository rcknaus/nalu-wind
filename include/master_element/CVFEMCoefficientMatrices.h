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

template <int p, typename Scalar>
struct CoefficientMatrices
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

template <int p> struct StaticCoefficientMatrices {};

//  const nodal_matrix_array<double, p> nodalWeights{coeffs::nodal_integration_weights<double, p>()};

//
//namespace static_coeffs {
//  constexpr LocalArray<double[2]> scsDeriv = {{-0.5, +0.5 }};
//}



//template <typename ArrayType, typename Enable = void> struct Data{};
//
//template <typename ArrayType>
//struct Data<ArrayType, typename std::enable_if<std::rank<ArrayType>::value == 1>::type>
//{
//  double internal_data_[2];
//
//  using value_type = typename std::remove_all_extents<ArrayType>::type;
//  KOKKOS_FORCEINLINE_FUNCTION value_type& operator()(int i) { return internal_data_[i]; }
//  KOKKOS_FORCEINLINE_FUNCTION  const value_type& operator()(int i) const { return internal_data_[i]; }
//};
//
//
//template <typename ArrayType>
//struct TestData { double data_[2]; };
//
//struct StaticCoeffs
//{
//  static constexpr TestData<double[2]> scsDeriv{{-0.5, +0.5 }};
//};


//namespace StaticCoeffs {
//  constexpr StaticLocalArray<double[2]> scsDeriv = {{-0.5, +0.5}};
//}

//template <> struct StaticCoefficientMatrices<1> {
//  static constexpr StaticLocalArray<double[2]> scsDeriv = {{-0.5, +0.5}};
//};


} // namespace naluUnit
} // namespace Sierra

#endif

