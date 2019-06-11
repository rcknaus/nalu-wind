/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "CVFEMCorrectedMassFlux.h"

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <element_promotion/NodeMapMaker.h>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu {

template <int p> ko::scs_scalar_view<p> corrected_mass_flux(
  double projTimeScale,
  ko::vector_view<p> coordinates,
  ko::scalar_view<p> rho,
  ko::vector_view<p> vel,
  ko::scalar_view<p> pressure,
  ko::vector_view<p> gradP)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto corrected_mass_flux = ko::scs_scalar_view<p>("corrected_mass_flux", coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto work_metric = scs_vector_array<DoubleType, p>();
    auto metric = la::make_view(work_metric);
    high_order_metrics::compute_laplacian_metric_linear(ops, local_coords, metric);

    auto local_rho = nodal_scalar_view<p,DoubleType>(&rho(index,0,0,0));
    auto local_Gp = nodal_vector_view<p,DoubleType>(&gradP(index,0,0,0,0));
    auto local_vel = nodal_vector_view<p,DoubleType>(&vel(index,0,0,0,0));
    auto local_pressure = nodal_scalar_view<p,DoubleType>(&pressure(index,0,0,0));

    auto work_mdot = LocalArray<DoubleType[3][p+1][p+1][p+1]>();
    auto mdot = scs_scalar_view<p,DoubleType>(work_mdot.data());
    high_order_metrics::compute_mdot_linear(
      ops,
      local_coords,
      metric,
      projTimeScale,
      local_rho,
      local_vel,
      local_Gp,
      local_pressure,
      mdot
    );

    for (int dj = 0; dj < 3; ++dj) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            corrected_mass_flux(index, dj, k, j, i) = mdot(dj, k, j, i);
          }
        }
      }
    }
  }
  return corrected_mass_flux;
}

template ko::scs_scalar_view<1> corrected_mass_flux<1>(
  double,
  ko::vector_view<1>,
  ko::scalar_view<1>,
  ko::vector_view<1>,
  ko::scalar_view<1>,
  ko::vector_view<1>);

template ko::scs_scalar_view<2> corrected_mass_flux<2>(
  double,
  ko::vector_view<2>,
  ko::scalar_view<2>,
  ko::vector_view<2>,
  ko::scalar_view<2>,
  ko::vector_view<2>);

template ko::scs_scalar_view<3> corrected_mass_flux<3>(
  double,
  ko::vector_view<3>,
  ko::scalar_view<3>,
  ko::vector_view<3>,
  ko::scalar_view<3>,
  ko::vector_view<3>);

template ko::scs_scalar_view<4> corrected_mass_flux<4>(
  double,
  ko::vector_view<4>,
  ko::scalar_view<4>,
  ko::vector_view<4>,
  ko::scalar_view<4>,
  ko::vector_view<4>);

}}
