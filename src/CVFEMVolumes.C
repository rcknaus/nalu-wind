/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "CVFEMVolumes.h"

#include "CVFEMTypeDefs.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "MatrixFreeTraits.h"

#include "master_element/TensorProductCVFEMVolumeMetric.h"
#include "element_promotion/NodeMapMaker.h"

namespace sierra { namespace nalu {

template <int p> ko::scalar_view<p> volumes(ko::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto volume = ko::scalar_view<p>("volume" + std::to_string(rand()), coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto work_vol = LocalArray<DoubleType[p+1][p+1][p+1]>();
    auto vol = la::make_view(work_vol);
    high_order_metrics::compute_volume_metric_linear(ops, local_coords, vol);

    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          volume(index, k, j, i) = vol(k, j, i);
        }
      }
    }
  }
  return volume;
}
template ko::scalar_view<POLY1> volumes<POLY1>(ko::vector_view<POLY1>);
template ko::scalar_view<POLY2> volumes<POLY2>(ko::vector_view<POLY2>);
template ko::scalar_view<POLY3> volumes<POLY3>(ko::vector_view<POLY3>);
template ko::scalar_view<POLY4> volumes<POLY4>(ko::vector_view<POLY4>);

template <int p> ko::scalar_view<p> volumes(ko::scalar_view<p> alpha, ko::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto volume = ko::scalar_view<p>("volume", coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto work_vol = LocalArray<DoubleType[p+1][p+1][p+1]>();
    auto vol = la::make_view(work_vol);
    high_order_metrics::compute_volume_metric_linear(ops, local_coords, vol);

    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          volume(index, k, j, i) = vol(k, j, i) * alpha(index, k, j, i);
        }
      }
    }
  }
  return volume;
}
template ko::scalar_view<POLY1> volumes<POLY1>(ko::scalar_view<POLY1>, ko::vector_view<POLY1>);
template ko::scalar_view<POLY2> volumes<POLY2>(ko::scalar_view<POLY2>, ko::vector_view<POLY2>);
template ko::scalar_view<POLY3> volumes<POLY3>(ko::scalar_view<POLY3>, ko::vector_view<POLY3>);
template ko::scalar_view<POLY4> volumes<POLY4>(ko::scalar_view<POLY4>, ko::vector_view<POLY4>);

}}
