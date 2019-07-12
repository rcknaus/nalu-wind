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
    ut_(1), mu_(1./550.), delta_(1.)
{}

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
  constexpr double correlation_factor = 1./0.09;
  constexpr double correlation_power = 1./0.88;
  const double Retau = ut_ * delta_ / mu_;
  const auto u_bulk_est = 0.5 * mu_ / delta_ * std::pow(correlation_factor * Retau, correlation_power);

  const double cos_integral = 4. / M_PI;
  const double u_m = u_bulk_est / cos_integral;

  const double x_center = M_PI;
  const double y_center = delta_;
  const double z_center = M_PI / 2;

  for(unsigned p=0; p < numPoints; ++p) {
    const double x = coords[0] - x_center;
    const double y = coords[1] - y_center;
    const double z = coords[2] - z_center;

    fieldPtr[0] =  1.0*u_m*cos(0.5*M_PI*y); //profile matches bc
    fieldPtr[1] =  0.1*u_m*cos(x)*cos(M_PI/2.0*y)*sin(2.0*z)*-2.0; // solenoidal comps
    fieldPtr[2] =  0.1*u_m*cos(x)*sin(M_PI/2.0*y)*cos(2.0*z)*0.5*M_PI;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
