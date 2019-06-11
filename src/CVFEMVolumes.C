/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <CVFEMVolumes.h>

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <element_promotion/NodeMapMaker.h>
#include <stk_util/util/ReportHandler.hpp>

namespace sierra { namespace nalu {
namespace {

int num_total_simd_elements(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector)
{
  int len = 0;
  const auto& buckets = bulk.get_buckets(stk::topology::ELEMENT_RANK, selector);
  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
  Kokkos::parallel_for(policy, [&len, buckets](const sierra::nalu::TeamHandleType& team)
  {
    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
    const size_t bucketLen = b.size();
    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&len] (int e) { Kokkos::atomic_add(&len, 1); });
  });
  return len;
}

int field_bucket_index(int bktIndex, int simdElemIndex) { return bktIndex * stk::simd::ndoubles + simdElemIndex; }

Kokkos::Array<stk::mesh::Entity, simdLen> load_simd_elems(const stk::mesh::Bucket& b, int bktIndex,  int numSimdElems)
{
  Kokkos::Array<stk::mesh::Entity, simdLen> simdElems;
  for (int simdElemIndex = 0; simdElemIndex < numSimdElems; ++simdElemIndex) {
    simdElems[simdElemIndex] = b[field_bucket_index(bktIndex, simdElemIndex)];
  }
  return simdElems;
}

}

template <int p> ko::scalar_view<p> volumes(ko::vector_view<p> coordinates)
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
          volume(index, k, j, i) = vol(k, j, i);
        }
      }
    }
  }
  return volume;
}
template ko::scalar_view<1> volumes<1>(ko::vector_view<1>);
template ko::scalar_view<2> volumes<2>(ko::vector_view<2>);
template ko::scalar_view<3> volumes<3>(ko::vector_view<3>);
template ko::scalar_view<4> volumes<4>(ko::vector_view<4>);

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
template ko::scalar_view<1> volumes<1>(ko::scalar_view<1>, ko::vector_view<1>);
template ko::scalar_view<2> volumes<2>(ko::scalar_view<2>, ko::vector_view<2>);
template ko::scalar_view<3> volumes<3>(ko::scalar_view<3>, ko::vector_view<3>);
template ko::scalar_view<4> volumes<4>(ko::scalar_view<4>, ko::vector_view<4>);

}}
