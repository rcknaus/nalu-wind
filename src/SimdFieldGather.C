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

int count_entities(const stk::mesh::BucketVector& buckets) {
  int num_scalar_items = 0;
  for (const auto* ib : buckets) {
    num_scalar_items += ib->size();
  }
  return num_scalar_items;
}

int num_simd_elements(const stk::mesh::BucketVector& buckets) {
  const int num_scalar_items = count_entities(buckets);
  const int remainder = num_scalar_items % simdLen;
  const auto add_element_for_remainder = (remainder > 0) ? 1 : 0;
  const int num_simd_elements = num_scalar_items / simdLen + add_element_for_remainder;
  return num_simd_elements;
}

int bucket_index(int bktIndex, int simdElemIndex) { return bktIndex * simdLen + simdElemIndex; }

template <typename Func> void iterate_buckets(const stk::mesh::BucketVector& buckets, Func f) {
  int mesh_index = 0;
  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
  Kokkos::parallel_for(policy, [&mesh_index, buckets, f](const sierra::nalu::TeamHandleType& team)
  {
    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
    const size_t bucketLen = b.size();
    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&b, &mesh_index, simdBucketLen, bucketLen, f] (int e)
    {
      const int numSimdElems = get_length_of_next_simd_group(e, bucketLen);
      for (int n = 0; n < numSimdElems; ++n) {
        f(mesh_index, n, b.begin_nodes(bucket_index(e, n)));
      }
      Kokkos::atomic_add(&mesh_index, 1);
    });
  });
}
}

template <int p> elem_ordinal_view_t<p> element_entity_offset_to_gid_map(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  Kokkos::View<int*> entToLidMap)
{
  auto perm = make_node_map_hex(p);
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);
  elem_ordinal_view_t<p> entityElemRowMap("elem_entity_row_map" + std::to_string(rand()), num_simd_elements(buckets));

  constexpr int n1D = p + 1;
  int localSimdMeshIndex = 0;
  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
  Kokkos::parallel_for(policy, [&](const TeamHandleType& team)
  {
    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
    const size_t bucketLen = b.size();
    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen),
      [&b, &localSimdMeshIndex, entToLidMap, perm, entityElemRowMap, simdBucketLen, bucketLen](int e) {
      const int numSimdElems = get_length_of_next_simd_group(e, bucketLen);
      for (int elemSimdIndex = 0; elemSimdIndex < simdLen; ++elemSimdIndex) {
        if (elemSimdIndex < numSimdElems)  {
          const auto& nodes = b.begin_nodes(bucket_index(e, elemSimdIndex));
          for (int k = 0; k < n1D; ++k) {
            for (int j = 0; j < n1D; ++j) {
              for (int i = 0; i < n1D; ++i) {
                entityElemRowMap(localSimdMeshIndex, elemSimdIndex, k, j, i) =  entToLidMap(nodes[perm(k, j, i)].local_offset());
              }
            }
          }
        }
        else {
          for (int k = 0; k < n1D; ++k) {
            for (int j = 0; j < n1D; ++j) {
              for (int i = 0; i < n1D; ++i) {
                entityElemRowMap(localSimdMeshIndex, elemSimdIndex, k, j, i) = invalid_node_index;
              }
            }
          }
        }
      }
      Kokkos::atomic_add(&localSimdMeshIndex, 1);
    });
  });
  return entityElemRowMap;
}
template elem_ordinal_view_t<POLY1> element_entity_offset_to_gid_map<POLY1>(const stk::mesh::BulkData&, const stk::mesh::Selector&, Kokkos::View<int*>);
template elem_ordinal_view_t<POLY2> element_entity_offset_to_gid_map<POLY2>(const stk::mesh::BulkData&, const stk::mesh::Selector&, Kokkos::View<int*>);
template elem_ordinal_view_t<POLY3> element_entity_offset_to_gid_map<POLY3>(const stk::mesh::BulkData&, const stk::mesh::Selector&, Kokkos::View<int*>);
template elem_ordinal_view_t<POLY4> element_entity_offset_to_gid_map<POLY4>(const stk::mesh::BulkData&, const stk::mesh::Selector&, Kokkos::View<int*>);

template <int p> elem_entity_view_t<p> element_entity_view(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector)
{
  auto perm = make_node_map_hex(p);
  const auto& buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector);
  elem_entity_view_t<p> entityElemView("elem_entity_row_map" + std::to_string(rand()), num_simd_elements(buckets));

  constexpr int n1D = p + 1;
  int localSimdMeshIndex = 0;
  auto policy = Kokkos::TeamPolicy<HostSpace>(buckets.size(), Kokkos::AUTO);
  Kokkos::parallel_for(policy, [&](const TeamHandleType& team)
  {
    const stk::mesh::Bucket& b = *buckets[team.league_rank()];
    const size_t bucketLen = b.size();
    const size_t simdBucketLen = get_num_simd_groups(bucketLen);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen),
      [&b, &localSimdMeshIndex, perm, entityElemView, simdBucketLen, bucketLen](int e) {
      const int numSimdElems = get_length_of_next_simd_group(e, bucketLen);
      for (int elemSimdIndex = 0; elemSimdIndex < simdLen; ++elemSimdIndex) {
        if (elemSimdIndex < numSimdElems)  {
          const auto& nodes = b.begin_nodes(bucket_index(e, elemSimdIndex));
          for (int k = 0; k < n1D; ++k) {
            for (int j = 0; j < n1D; ++j) {
              for (int i = 0; i < n1D; ++i) {
                entityElemView(localSimdMeshIndex, elemSimdIndex, k, j, i) = nodes[perm(k, j, i)];
              }
            }
          }
        }
        else {
          for (int k = 0; k < n1D; ++k) {
            for (int j = 0; j < n1D; ++j) {
              for (int i = 0; i < n1D; ++i) {
                entityElemView(localSimdMeshIndex, elemSimdIndex, k, j, i) = stk::mesh::Entity(stk::mesh::Entity::InvalidEntity);
              }
            }
          }
        }
      }
      Kokkos::atomic_add(&localSimdMeshIndex, 1);
    });
  });
  return entityElemView;
}
template elem_entity_view_t<POLY1> element_entity_view<POLY1>(const stk::mesh::BulkData&, const stk::mesh::Selector&);
template elem_entity_view_t<POLY2> element_entity_view<POLY2>(const stk::mesh::BulkData&, const stk::mesh::Selector&);
template elem_entity_view_t<POLY3> element_entity_view<POLY3>(const stk::mesh::BulkData&, const stk::mesh::Selector&);
template elem_entity_view_t<POLY4> element_entity_view<POLY4>(const stk::mesh::BulkData&, const stk::mesh::Selector&);

template <int p> void write_to_stk_field(
  const stk::mesh::BulkData& bulk,
  const elem_ordinal_view_t<p> entToLID,
  const elem_view::scalar_view<p> element_field,
  ScalarFieldType& nodal_stk_field)
{
  for (int index = 0; index < entToLID.extent_int(0); ++index) {
    for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            const auto node = bulk.get_entity(stk::topology::NODE_RANK, entToLID(index,n,k,j,i)+1);
            *stk::mesh::field_data(nodal_stk_field, node) = stk::simd::get_data(element_field(index,k,j,i), n);
          }
        }
      }
    }
  }
}
template void write_to_stk_field<POLY1>(const stk::mesh::BulkData&, const elem_ordinal_view_t<POLY1>, const elem_view::scalar_view<POLY1>, ScalarFieldType&);
template void write_to_stk_field<POLY2>(const stk::mesh::BulkData&, const elem_ordinal_view_t<POLY2>, const elem_view::scalar_view<POLY2>, ScalarFieldType&);
template void write_to_stk_field<POLY3>(const stk::mesh::BulkData&, const elem_ordinal_view_t<POLY3>, const elem_view::scalar_view<POLY3>, ScalarFieldType&);
template void write_to_stk_field<POLY4>(const stk::mesh::BulkData&, const elem_ordinal_view_t<POLY4>, const elem_view::scalar_view<POLY4>, ScalarFieldType&);

template <int p> elem_view::scalar_view<p> gather_field(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, const ScalarFieldType& field)
{
  auto perm = make_node_map_hex(p);
  const auto& buckets = bulk.get_buckets(stk::topology::ELEMENT_RANK, stk::mesh::selectField(field) & selector);
  auto scalar_field_view = elem_view::scalar_view<p>(field.name() + "_view" + std::to_string(rand()), num_simd_elements(buckets));
  iterate_buckets(buckets, [&scalar_field_view, &field, perm] (int simdElemIndex, int localSimdIndex, const stk::mesh::Entity* nodes)
  {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          auto val = *stk::mesh::field_data(field, nodes[perm(k,j,i)]);
          stk::simd::set_data(scalar_field_view(simdElemIndex, k, j, i), localSimdIndex, val);
        }
      }
    }
  });
  return scalar_field_view;
}
template elem_view::scalar_view<POLY1> gather_field<POLY1>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
template elem_view::scalar_view<POLY2> gather_field<POLY2>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
template elem_view::scalar_view<POLY3> gather_field<POLY3>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);
template elem_view::scalar_view<POLY4> gather_field<POLY4>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const ScalarFieldType&);

template <int p> elem_view::vector_view<p> gather_field(const stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector, const VectorFieldType& field)
{
  auto perm = make_node_map_hex(p);
  const auto& buckets = bulk.get_buckets(stk::topology::ELEMENT_RANK, stk::mesh::selectField(field) & selector);
  auto vector_field_view = elem_view::vector_view<p>(field.name() + "_view" + std::to_string(rand()), num_simd_elements(buckets));
  iterate_buckets(buckets, [&vector_field_view, &field, perm] (int simdElemIndex, int localSimdIndex, const stk::mesh::Entity* nodes)
  {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          const auto* val = stk::mesh::field_data(field, nodes[perm(k, j, i)]);
          for (int d = 0; d < 3; ++ d) {
            stk::simd::set_data(vector_field_view(simdElemIndex, k, j, i, d), localSimdIndex, val[d]);
          }
        }
      }
    }
  });
  return vector_field_view;
}
template elem_view::vector_view<POLY1> gather_field<POLY1>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);
template elem_view::vector_view<POLY2> gather_field<POLY2>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);
template elem_view::vector_view<POLY3> gather_field<POLY3>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);
template elem_view::vector_view<POLY4> gather_field<POLY4>(const stk::mesh::BulkData&, const stk::mesh::Selector&, const VectorFieldType&);

}}
