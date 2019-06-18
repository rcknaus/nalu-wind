/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level NaluUnit      */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef HighOrderCoefficients_h
#define HighOrderCoefficients_h

#include <element_promotion/QuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <CVFEMTypeDefs.h>

#include <stk_util/util/ReportHandler.hpp>

namespace sierra {
namespace nalu {

namespace coeffs {

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> nodal_integration_weights(const double* /* nodeLocs */, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  auto scsEndLoc = pad_end_points(p, scsLocs);
  auto weightvec = SGL_quadrature_rule(nodes1D, scsEndLoc.data()).second;

  nodal_coefficient_matrix<Scalar, p> weights;
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      weights(j, i) = weightvec[j * nodes1D + i];
    }
  }
  return weights;
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> lumped_nodal_integration_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  const auto weights = nodal_integration_weights<Scalar, p>(nodeLocs, scsLocs);

  Kokkos::Array<Scalar, p + 1> lumped_weights;
  for (int j = 0; j < p+1; ++j) {
    lumped_weights[j] = 0;
  }
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      lumped_weights[j] += weights(j,i);
    }
  }
  nodal_coefficient_matrix<Scalar, p> lumped_weight_matrix;

  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      lumped_weight_matrix(j,i) = (i == j) ? lumped_weights[j] : 0;
    }
  }
  return lumped_weight_matrix;
}

template <typename Scalar, int p>
scs_matrix_array<Scalar, p> scs_interpolation_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  scs_matrix_array<Scalar, p> scsInterp;

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      scsInterp(j,i) = basis1D.interpolation_weight(scsLocs[j], i);
    }
  }
  return scsInterp;
}

template <typename Scalar, int p>
scs_matrix_array<Scalar, p> scs_derivative_weights(const double* nodeLocs, const double* scsLocs)
{
  constexpr int nodes1D = p+1;
  scs_matrix_array<Scalar, p> scsDeriv;

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < p; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      scsDeriv(j,i) = basis1D.derivative_weight(scsLocs[j], i);
    }
  }
  return scsDeriv;
}

template<typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> nodal_derivative_weights(const double* nodeLocs)
{
  constexpr int nodes1D = p + 1;
  nodal_coefficient_matrix<Scalar, p> nodalDeriv;

  auto basis1D = Lagrange1D(nodeLocs, p);
  for (int j = 0; j < nodes1D; ++j) {
    for (int i = 0; i < nodes1D; ++i) {
      nodalDeriv(j, i) = basis1D.derivative_weight(nodeLocs[j], i);
    }
  }
  return nodalDeriv;
}

template<typename Scalar, int p>
linear_scs_matrix_array<Scalar, p> linear_scs_interpolation_weights(const double* scsLocs)
{
  linear_scs_matrix_array<Scalar, p> linear_scs_interp;

  for (int j = 0; j < p; ++j)  {
    linear_scs_interp(0, j) = 0.5 * (1 - scsLocs[j]);
    linear_scs_interp(1, j) = 0.5 * (1 + scsLocs[j]);
  }

  return linear_scs_interp;
}

template<typename Scalar, int p>
linear_nodal_matrix_array<Scalar, p> linear_nodal_interpolation_weights(const double* nodeLocs)
{
  linear_nodal_matrix_array<Scalar, p> linear_nodal_interp;

  for (int j = 0; j < p + 1; ++j)
  {
    linear_nodal_interp(0, j) = 0.5 * (1 - nodeLocs[j]);
    linear_nodal_interp(1, j) = 0.5 * (1 + nodeLocs[j]);
  }
  return linear_nodal_interp;
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> nodal_integration_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs = gauss_legendre_rule(p).first;

  return nodal_integration_weights<Scalar, p>(nodeLocs.data(), scsLocs.data());
}


template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> lumped_nodal_integration_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return lumped_nodal_integration_weights<Scalar, p>(nodeLocs.data(), scsLocs.data());
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> nodal_derivative_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  return nodal_derivative_weights<Scalar, p>(nodeLocs.data());
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> scs_derivative_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return scs_derivative_weights<Scalar, p>(nodeLocs.data(), scsLocs.data());
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> scs_interpolation_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  auto scsLocs  = gauss_legendre_rule(p).first;

  return scs_interpolation_weights<Scalar, p>(nodeLocs.data(), scsLocs.data());
}

template <typename Scalar, int p>
linear_scs_matrix_array<Scalar, p> linear_scs_interpolation_weights()
{
  auto scsLocs  = gauss_legendre_rule(p).first;
  return linear_scs_interpolation_weights<Scalar, p>(scsLocs.data());
}

template <typename Scalar, int p>
linear_nodal_matrix_array<Scalar, p> linear_nodal_interpolation_weights()
{
  auto nodeLocs = gauss_lobatto_legendre_rule(p+1).first;
  return linear_nodal_interpolation_weights<Scalar, p>(nodeLocs.data());
}

template <typename Scalar, int p>
nodal_coefficient_matrix<Scalar, p> difference_matrix()
{
  auto scatt = la::zero<nodal_coefficient_matrix<Scalar, p>>();
  scatt(0, 0)  = -1;
  scatt(p,p-1) = +1;

  for (int j = 1; j < p; ++j) {
    scatt(j,j+0) = -1;
    scatt(j,j-1) = +1;
  }
  return scatt;
}

template <typename Scalar, int p>
nodal_matrix_array<Scalar, p> identity_matrix()
{
  auto id = la::zero<nodal_coefficient_matrix<Scalar, p>>();
  for (int j = 0; j < p+1; ++j) {
    id(j,j) = 1.0;
  }

  return id;
}

} // namespace CoefficientMatrices
} // namespace naluUnit
} // namespace Sierra

#endif
