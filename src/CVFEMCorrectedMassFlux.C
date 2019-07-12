/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "CVFEMCorrectedMassFlux.h"

#include "MatrixFreeTraits.h"
#include <CVFEMTypeDefs.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu {

template <int p> elem_view::scs_scalar_view<p> corrected_mass_flux(
  double projTimeScale,
  elem_view::vector_view<p> coordinates,
  elem_view::scalar_view<p> rho,
  elem_view::vector_view<p> vel,
  elem_view::scalar_view<p> pressure,
  elem_view::vector_view<p> gradP)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto corrected_mass_flux = elem_view::scs_scalar_view<p>("corrected_mass_flux" + std::to_string(rand()), coordinates.extent_int(0));
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

template elem_view::scs_scalar_view<POLY1> corrected_mass_flux<POLY1>(
  double,
  elem_view::vector_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::vector_view<POLY1>,
  elem_view::scalar_view<POLY1>,
  elem_view::vector_view<POLY1>);

template elem_view::scs_scalar_view<POLY2> corrected_mass_flux<POLY2>(
  double,
  elem_view::vector_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::vector_view<POLY2>,
  elem_view::scalar_view<POLY2>,
  elem_view::vector_view<POLY2>);

template elem_view::scs_scalar_view<POLY3> corrected_mass_flux<POLY3>(
  double,
  elem_view::vector_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::vector_view<POLY3>,
  elem_view::scalar_view<POLY3>,
  elem_view::vector_view<POLY3>);

template elem_view::scs_scalar_view<POLY4> corrected_mass_flux<POLY4>(
  double,
  elem_view::vector_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::vector_view<POLY4>,
  elem_view::scalar_view<POLY4>,
  elem_view::vector_view<POLY4>);

}}
