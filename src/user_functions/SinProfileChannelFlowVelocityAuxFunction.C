// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/SinProfileChannelFlowVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

SinProfileChannelFlowVelocityAuxFunction::
  SinProfileChannelFlowVelocityAuxFunction(
    const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos)
{
  // does nothing
}

double
reichardt(double y)
{
  constexpr double kappa = 0.41;
  return (
    1. / kappa * std::log(1 + kappa * y) +
    7.8 * (1 - std::exp(-y / 11.) - y / 11. * std::exp(-y / 3.)));
}

void
SinProfileChannelFlowVelocityAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  constexpr double ut = 1.0;
  constexpr double retau = 2524.0;
  constexpr double delta = 1.0;
  constexpr double mu = 1. / retau;
  constexpr double correlation_factor = 1. / 0.09;
  constexpr double correlation_power = 1. / 0.88;
  const auto u_bulk_est =
    0.5 * mu / delta * std::pow(correlation_factor * retau, correlation_power);

  const double cos_integral = 4. / M_PI;
  const double u_0 = 0.1 * u_bulk_est / cos_integral;

  const double x_center = 0;
  const double y_center = 0;
  const double z_center = 0;

  constexpr double a = M_PI * 2;
  constexpr double b = M_PI * 8;
  constexpr double c = M_PI * 4;

  for (unsigned p = 0; p < numPoints; ++p) {
    const double x = coords[0] - x_center;
    const double y = coords[1] - y_center;
    const double z = coords[2] - z_center;

    const double yplus =
      (y < 0) ? (delta + y) * ut / mu : (delta - y) * ut / mu;
    fieldPtr[0] = reichardt(yplus);
    fieldPtr[1] = u_0 * std::cos(c * x) * std::cos(b * z) * std::sin(a * y);
    fieldPtr[2] = u_0 * std::cos(c * x) * std::sin(a * y);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
