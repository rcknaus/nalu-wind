/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <CVFEMMappedAreas.h>

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <element_promotion/NodeMapMaker.h>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu {

template <int p> ko::scs_vector_view<p> mapped_areas(ko::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto mapped_area = ko::scs_vector_view<p>("mapped_area", coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto work_metric = scs_vector_array<DoubleType, p>();
    auto metric = la::make_view(work_metric);
    high_order_metrics::compute_laplacian_metric_linear(ops, local_coords, metric);

    for (int dj = 0; dj < 3; ++dj) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int di = 0; di < 3; ++di) {
              mapped_area(index, dj, k, j, i, di) = metric(dj, k, j, i, di);
            }
          }
        }
      }
    }
  }
  return mapped_area;
}
template ko::scs_vector_view<1> mapped_areas<1>(ko::vector_view<1>);
template ko::scs_vector_view<2> mapped_areas<2>(ko::vector_view<2>);
template ko::scs_vector_view<3> mapped_areas<3>(ko::vector_view<3>);
template ko::scs_vector_view<4> mapped_areas<4>(ko::vector_view<4>);

template <int p> ko::scs_vector_view<p> mapped_areas(ko::scalar_view<p> alpha, ko::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto mapped_area = ko::scs_vector_view<p>("mapped_area", coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto local_alpha = nodal_scalar_view<p, DoubleType>(&alpha(index,0,0,0));
    auto work_metric = LocalArray<DoubleType[3][p+1][p+1][p+1][3]>();
    auto metric = la::make_view(work_metric);
    high_order_metrics::compute_diffusion_metric_linear(ops, local_coords, local_alpha, metric);

    for (int dj = 0; dj < 3; ++dj) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int di = 0; di < 3; ++di) {
              mapped_area(index, dj, k, j, i, di) = metric(dj, k, j, i, di);
            }
          }
        }
      }
    }
  }
  return mapped_area;
}
template ko::scs_vector_view<1> mapped_areas<1>(ko::scalar_view<1>, ko::vector_view<1>);
template ko::scs_vector_view<2> mapped_areas<2>(ko::scalar_view<2>, ko::vector_view<2>);
template ko::scs_vector_view<3> mapped_areas<3>(ko::scalar_view<3>, ko::vector_view<3>);
template ko::scs_vector_view<4> mapped_areas<4>(ko::scalar_view<4>, ko::vector_view<4>);
}}
