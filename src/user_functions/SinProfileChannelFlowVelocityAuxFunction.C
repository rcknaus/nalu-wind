/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/SinProfileChannelFlowVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

SinProfileChannelFlowVelocityAuxFunction::SinProfileChannelFlowVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) :
  AuxFunction(beginPos, endPos),
  u_m(1)
{}


double reichardt(double y)
{
  constexpr double kappa = 0.41;
  return (1./kappa*std::log(1+kappa*y) + 7.8*(1-std::exp(-y/11.) - y/11.*std::exp(-y/3.)));
}

void
SinProfileChannelFlowVelocityAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  constexpr double ut = 1.0;
  constexpr double mu = 1.0/590.0;
  constexpr double delta  = 1.0;
  constexpr double correlation_factor = 1./0.09;
  constexpr double correlation_power = 1./0.88;
  const double Retau = ut * delta / mu;
  const auto u_bulk_est = 0.5 * mu / delta * std::pow(correlation_factor * Retau, correlation_power);

  const double cos_integral = 4. / M_PI;
  const double u_0 = u_bulk_est / cos_integral;

  const double x_center = 0; //M_PI;
  const double y_center = 0; //delta_;
  const double z_center = 0; //M_PI / 2;

  constexpr double a = 3/2.;
  constexpr double b = 8;
  constexpr double c = 4;

  for(unsigned p=0; p < numPoints; ++p) {
    const double x = coords[0] - x_center;
    const double y = coords[1] - y_center;
    const double z = coords[2] - z_center;

    const double yplus = (y < 0) ? (delta+y)*ut/mu : (delta-y)*ut/mu;
    fieldPtr[0] =  reichardt(yplus);
    fieldPtr[1] =  0.1*u_0*cos(c*x)*cos(M_PI*y*a) * sin(b*z) * (-M_PI*a);
    fieldPtr[2] =  0.1*u_0*cos(c*x)*cos(M_PI*y*a) * cos(b*z) * b;


    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
