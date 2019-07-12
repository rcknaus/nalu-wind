/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "SimdFieldGather.h"
#include "MatrixFreeTraits.h"
#include "CVFEMTypeDefs.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "master_element/TensorProductCVFEMVolumeMetric.h"
#include "element_promotion/NodeMapMaker.h"
#include "MatrixFreeTraits.h"

namespace sierra { namespace nalu{

namespace {

int count_entities(const stk::mesh::BucketVector& buckets)
{
  int num_scalar_items = 0;
  for (const auto* ib : buckets) {
    num_scalar_items += ib->size();
  }
  return num_scalar_items;
}


template <typename Func> void iterate_node_buckets(const stk::mesh::BucketVector& buckets, Func f) {
  int mesh_index = 0;
  for (const auto* ib : buckets) {
    for (const auto entity : *ib) {
      f(mesh_index, entity);
      ++mesh_index;
    }
  }
}
}

Kokkos::View<int*> node_offset_view(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, Kokkos::View<int*> entToLid)
{
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
  auto nodeOffsets = Kokkos::View<int*>("node_offset_view" + std::to_string(rand()), count_entities(buckets));

  if (nodeOffsets.extent_int(0) == 0) { return nodeOffsets; }

  int local_index = 0;
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      nodeOffsets(local_index) = entToLid(node.local_offset());
      local_index++;
    }
  }
  return nodeOffsets;
}


node_view::scalar_view<> gather_node_field(const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector, const ScalarFieldType& field)
{
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectField(field) & selector);
  auto scalar_field_view = node_view::scalar_view<>(field.name() + "_view" + std::to_string(rand()), count_entities(buckets));
  iterate_node_buckets(buckets, [scalar_field_view, &field] (int meshIndex, stk::mesh::Entity node)
  {
    scalar_field_view(meshIndex) = *stk::mesh::field_data(field, node);
  });
  return scalar_field_view;
}

node_view::vector_view<> gather_node_field(const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector, const VectorFieldType& field)
{
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectField(field) & selector);
  auto vector_field_view = node_view::vector_view<>(field.name() + "_view" + std::to_string(rand()), count_entities(buckets));

  iterate_node_buckets(buckets, [vector_field_view, &field] (int meshIndex, stk::mesh::Entity node)
  {
    const auto* data = stk::mesh::field_data(field, node);
    vector_field_view(meshIndex, 0) = data[0];
    vector_field_view(meshIndex, 1) = data[1];
    vector_field_view(meshIndex, 2) = data[2];
  });
  return vector_field_view;
}

//namespace {
//template <typename Func> void iterate_node_buckets(const stk::mesh::BucketVector& buckets, Func f) {
//  int mesh_index = 0;
//  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
//  Kokkos::parallel_for(policy, [&mesh_index, buckets, f](const sierra::nalu::TeamHandleType& team)
//  {
//    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
//    const size_t bucketLen = b.size();
//    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&b, &mesh_index, simdBucketLen, bucketLen, f] (int k)
//    {
//      const int numSimdElems = get_length_of_next_simd_group(k, bucketLen);
//      for (int n = 0; n < numSimdElems; ++n) {
//        f(mesh_index, n, b[bucket_index(k, n)]);
//      }
//      Kokkos::atomic_add(&mesh_index, 1);
//    });
//  });
//}
//
//}
//node_offset_view_t node_offset_view(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector,
//  Kokkos::View<int*> entToLidMap)
//{
//  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, selector);
//  node_offset_view_t nodeOffsetView("node_entity_row_map" + std::to_string(rand()), num_simd_entities(buckets));
//  int localSimdMeshIndex = 0;
//  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
//  Kokkos::parallel_for(policy, [&](const TeamHandleType& team)
//  {
//    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
//    const size_t bucketLen = b.size();
//    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen),
//      [&b, &localSimdMeshIndex, entToLidMap, nodeOffsetView, simdBucketLen, bucketLen] (int k) {
//      const int numSimdElems = get_length_of_next_simd_group(k, bucketLen);
//      for (int nodeSimdIndex = 0; nodeSimdIndex < simdLen; ++nodeSimdIndex) {
//        if (nodeSimdIndex < numSimdElems)  {
//          const auto node = b[bucket_index(k, nodeSimdIndex)];
//          nodeOffsetView(localSimdMeshIndex, nodeSimdIndex) = entToLidMap(node.local_offset());
//        }
//        else {
//          nodeOffsetView(localSimdMeshIndex, nodeSimdIndex) = invalid_node_index;
//        }
//      }
//      ++localSimdMeshIndex;
////      Kokkos::atomic_add(&localSimdMeshIndex, 1);
//    });
//  });
//  return nodeOffsetView;
//}
//
//node_view::scalar_view<> gather_node_field(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, const ScalarFieldType& field)
//{
//  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectField(field) & selector);
//  auto scalar_field_view = node_view::scalar_view<>(field.name() + "_view" + std::to_string(rand()), num_simd_entities(buckets));
//  for (int k = 0; k < scalar_field_view.extent_int(0); ++k){
//    for (int d =  0; d < 3; ++d) {
//      scalar_field_view(k) = 0.;
//    }
//  }
//
//  iterate_node_buckets(buckets, [scalar_field_view, &field] (int simdElemIndex, int localSimdIndex, stk::mesh::Entity node)
//  {
//    stk::simd::set_data(scalar_field_view(simdElemIndex), localSimdIndex, *stk::mesh::field_data(field, node));
//  });
//  return scalar_field_view;
//}
//
//node_view::vector_view<> gather_node_field(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, const VectorFieldType& field)
//{
//  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, stk::mesh::selectField(field) & selector);
//  auto vector_field_view = node_view::vector_view<>(field.name() + "_view" + std::to_string(rand()), num_simd_entities(buckets));
//  for (int k = 0; k < vector_field_view.extent_int(0); ++k){
//    for (int d =  0; d < 3; ++d) {
//      vector_field_view(k, d) = 0.;
//    }
//  }
//
//  iterate_node_buckets(buckets, [vector_field_view, &field] (int simdElemIndex, int localSimdIndex, stk::mesh::Entity node)
//  {
//    const auto* vecFieldData = stk::mesh::field_data(field, node);
//    for (int d = 0; d < 3; ++d) {
//      stk::simd::set_data(vector_field_view(simdElemIndex, d), localSimdIndex, vecFieldData[d]);
//    }
//  });
//  return vector_field_view;
//}
}}
