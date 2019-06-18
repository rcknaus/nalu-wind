
#include "SparsifiedLaplacian.h"
#include "kernel/SparsifiedEdgeLaplacian.h"

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"
#include "MatrixFreeTraits.h"

#include "TpetraLinearSystem.h"

namespace sierra { namespace nalu {

template <int p>
SparsifiedLaplacianInterior<p>::SparsifiedLaplacianInterior(
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector,
  TpetraLinearSystem& linsys,
  const VectorFieldType& coordField)
: bulk_(bulk), selector_(selector), linsys_(linsys), coordField_(coordField)
{
  entities_ = element_entity_view<p>(bulk, selector);
  coords_ = gather_field<p>(bulk, selector, coordField);
}

template <int p> void SparsifiedLaplacianInterior<p>::initialize_connectivity()
{
  linsys_.buildSparsifiedElemToNodeGraph(selector_);
}

namespace {
template <typename ViewType, typename Scalar> KOKKOS_FORCEINLINE_FUNCTION
void hex_vertex_coordinates(int k, int j, int i,  const ViewType& xc, Scalar base_box[3][8])
{
  static_assert(ViewType::Rank == 4,"");
  for (int d = 0; d < 3; ++d) {
    base_box[d][0] = xc(k+0, j+0, i+0, d);
    base_box[d][1] = xc(k+0, j+0, i+1, d);
    base_box[d][2] = xc(k+0, j+1, i+1, d);
    base_box[d][3] = xc(k+0, j+1, i+0, d);
    base_box[d][4] = xc(k+1, j+0, i+0, d);
    base_box[d][5] = xc(k+1, j+0, i+1, d);
    base_box[d][6] = xc(k+1, j+1, i+1, d);
    base_box[d][7] = xc(k+1, j+1, i+0, d);
  }
}
}

template <int p> void SparsifiedLaplacianInterior<p>::compute_lhs_simd()
{
  std::vector<stk::mesh::Entity> entities(8, stk::mesh::Entity());
  std::vector<int> scratchIds(8, 0);
  std::vector<double> scratchVals(8, 0.0);

  std::vector<double> rhs(8, 0.0);
  std::vector<double> lhs(64, 0.0);
  static constexpr int perm[8] = {0, 1, 3, 2, 4, 5, 7, 6};

//  Kokkos::

  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    auto elem_coords = nodal_vector_view<p>(&coords_(index,0,0,0,0));
    auto all_lhs = sparsified_laplacian_lhs<p>(elem_coords);
    for (int nsimd= 0; nsimd < simdLen && entities_(index, nsimd, 0,0,0).is_local_offset_valid(); ++nsimd) {
      for (int n = 0; n < p; ++n) {
        for (int m = 0; m < p; ++m) {
          for (int l = 0; l < p; ++l) {
            entities[0] = entities_(index, nsimd, n + 0, m + 0, l + 0);
            entities[1] = entities_(index, nsimd, n + 0, m + 0, l + 1);
            entities[2] = entities_(index, nsimd, n + 0, m + 1, l + 1);
            entities[3] = entities_(index, nsimd, n + 0, m + 1, l + 0);
            entities[4] = entities_(index, nsimd, n + 1, m + 0, l + 0);
            entities[5] = entities_(index, nsimd, n + 1, m + 0, l + 1);
            entities[6] = entities_(index, nsimd, n + 1, m + 1, l + 1);
            entities[7] = entities_(index, nsimd, n + 1, m + 1, l + 0);
            for (int j = 0; j < 8; ++j) {
              for (int i = 0; i < 8; ++i) {
                lhs[8 * j + i] = stk::simd::get_data(all_lhs(n, m, l, perm[j], perm[i]), nsimd);
              }
            }
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
          }
        }
      }
    }
  }
}

namespace {
template <typename LATYPE>
void write_edge_lhs(int n, int m, int l, int edge_ord, int nsimd, const LATYPE& all_lhs, double* lhs)
{
  lhs[0] = stk::simd::get_data(all_lhs(n, m, l, edge_ord, 0, 0), nsimd);
  lhs[1] = stk::simd::get_data(all_lhs(n, m, l, edge_ord, 0, 1), nsimd);
  lhs[2] = stk::simd::get_data(all_lhs(n, m, l, edge_ord, 1, 0), nsimd);
  lhs[3] = stk::simd::get_data(all_lhs(n, m, l, edge_ord, 1, 1), nsimd);
}
}

template <typename EntType>
stk::mesh::Entity entity_for_base_node_ordinal(int index, int nsimd, int n, int m, int l, int node_ordinal, const EntType& ent)
{
  switch (node_ordinal)
  {
    case 0:  return ent(index, nsimd, n+0,m+0,l+0);
    case 1:  return ent(index, nsimd, n+0,m+0,l+1);
    case 2:  return ent(index, nsimd, n+0,m+1,l+1);
    case 3:  return ent(index, nsimd, n+0,m+1,l+0);
    case 4:  return ent(index, nsimd, n+1,m+0,l+0);
    case 5:  return ent(index, nsimd, n+1,m+0,l+1);
    case 6:  return ent(index, nsimd, n+1,m+1,l+1);
    case 7:  return ent(index, nsimd, n+1,m+1,l+0);
    default: return stk::mesh::Entity(stk::mesh::Entity::InvalidEntity);
  }
}

template <int p> void SparsifiedLaplacianInterior<p>::compute_lhs_simd_edge()
{
  std::vector<stk::mesh::Entity> entities(2, stk::mesh::Entity());
  std::vector<int> scratchIds(2, 0);
  std::vector<double> scratchVals(2, 0.0);

  std::vector<double> rhs(2, 0.0);
  std::vector<double> lhs(4, 0.0);

  static constexpr int edge_perm[12] = {
      0, 2, 4, 6,
      3, 1, 7, 5,
      8, 9, 11, 10
  };

  static constexpr int edges[12][2] =
  {
      {0,1}, //0
      {1,2}, //1
      {2,3}, //2
      {3,0}, //3 bottom face
      {4,5}, //4
      {5,6}, //5
      {6,7}, //6
      {7,4}, //7 top face
      {0,4}, //8
      {1,5}, //9
      {2,6}, //10
      {3,7}  //11 bottom-to-top
  };

  static constexpr int orient[12] = {
      +1, -1, +1, -1,
      -1, +1, -1, +1,
      +1, +1, +1, +1
  };

  for (int index  = 0; index < entities_.extent_int(0); ++index) {
    auto elem_coords = nodal_vector_view<p>(&coords_(index,0,0,0,0));
    auto all_lhs = sparsified_laplacian_edge_lhs<p>(elem_coords);
    for (int nsimd= 0; nsimd < simdLen && entities_(index, nsimd, 0,0,0).is_local_offset_valid(); ++nsimd) {
      for (int n = 0; n < p; ++n) {
        for (int m = 0; m < p; ++m) {
          for (int l = 0; l < p; ++l) {
            for (int current_ord = 0; current_ord < 12; ++current_ord) {
              const int leftNode =  (orient[current_ord] > 0) ? 0 : 1;
              const int rightNode = (orient[current_ord] > 0) ? 1 : 0;
              entities[0] = entity_for_base_node_ordinal(index, nsimd, n, m, l,  edges[edge_perm[current_ord]][leftNode], entities_);
              entities[1] = entity_for_base_node_ordinal(index, nsimd, n, m, l, edges[edge_perm[current_ord]][rightNode], entities_);
              write_edge_lhs(n, m, l, current_ord, nsimd, all_lhs, lhs.data());
              linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
            }

//            int current_ord = 0;
//            int edge_ord = edge_perm[current_ord];
//            int ln = (orient[current_ord] > 0) ? 0 : 1;
//            int rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entity_for_base_node_ordinal(index, nsimd, n, m, l, 0); // 0
//            entities[rn] = entities_(index, nsimd, n+0,m+0,l+1); // 1
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 1;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+0,l+1); // 1
//            entities[rn] = entities_(index, nsimd, n+0,m+1,l+1); // 2
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 2;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+1,l+1); // 2
//            entities[rn] = entities_(index, nsimd, n+0,m+1,l+0); // 3
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 3;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+1,l+0); // 3
//            entities[rn] = entities_(index, nsimd, n+0,m+0,l+0); // 0
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 4;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+1,m+0,l+0); // 4
//            entities[rn] = entities_(index, nsimd, n+1,m+0,l+1); // 5
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 5;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+1,m+0,l+1); // 5
//            entities[rn] = entities_(index, nsimd, n+1,m+1,l+1); // 6
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 6;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+1,m+1,l+1); // 6
//            entities[rn] = entities_(index, nsimd, n+1,m+1,l+0); // 7
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 7;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+1,m+1,l+0); // 7
//            entities[rn] = entities_(index, nsimd, n+1,m+0,l+0); // 4
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 8;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+0,l+0); // 0
//            entities[rn] = entities_(index, nsimd, n+1,m+0,l+0); // 4
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 9;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+0,l+1); // 1
//            entities[rn] = entities_(index, nsimd, n+1,m+0,l+1); // 5
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 10;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+1,l+1); // 2
//            entities[rn] = entities_(index, nsimd, n+1,m+1,l+1); // 6
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
//
//            current_ord = 11;
//            edge_ord = edge_perm[current_ord];
//            ln = (orient[current_ord] > 0) ? 0 : 1;
//            rn = (orient[current_ord] > 0) ? 1 : 0;
//            entities[ln] = entities_(index, nsimd, n+0,m+1,l+0); // 3
//            entities[rn] = entities_(index, nsimd, n+1,m+1,l+0); // 7
//            write_edge_lhs(n, m, l, edge_ord, nsimd, all_lhs, lhs.data());
//            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
          }
        }
      }
    }
  }
}

template <int p> void SparsifiedLaplacianInterior<p>::compute_lhs()
{
  std::vector<stk::mesh::Entity> entities(8, stk::mesh::Entity());
  std::vector<int> scratchIds(8, 0);
  std::vector<double> scratchVals(8, 0.0);

  std::vector<double> rhs(8, 0.0);
  std::vector<double> lhs(64, 0.0);

  const auto node_map = make_node_map_hex(p, true);
  const auto& buckets = bulk_.get_buckets(stk::topology::ELEM_RANK, selector_);
  for (const auto* ib : buckets) {
    const auto& b = *ib;
    for (size_t k = 0u; k < b.size(); ++k) {
      const auto* nodes = b.begin_nodes(k);
      for (int n = 0; n < p; ++n) {
        for (int m = 0; m < p; ++m) {
          for (int l = 0; l < p; ++l) {
            entities[0] = nodes[node_map(n+0,m+0,l+0)];
            entities[1] = nodes[node_map(n+0,m+0,l+1)];
            entities[2] = nodes[node_map(n+0,m+1,l+1)];
            entities[3] = nodes[node_map(n+0,m+1,l+0)];
            entities[4] = nodes[node_map(n+1,m+0,l+0)];
            entities[5] = nodes[node_map(n+1,m+0,l+1)];
            entities[6] = nodes[node_map(n+1,m+1,l+1)];
            entities[7] = nodes[node_map(n+1,m+1,l+0)];

            double box[3][8];
            for (int d = 0; d < 3; ++d) {
              for (int node = 0; node < 8; ++node) {
                box[d][node] =  stk::mesh::field_data(coordField_, entities[node])[d];
              }
            }
            auto local_lhs = laplacian_lhs(box);
            for (int j = 0; j < 8; ++j) {
              for (int i = 0; i < 8; ++i) {
                lhs[8 * j + i] = local_lhs(j,i);
              }
            }
            linsys_.sumInto(entities, scratchIds, scratchVals, rhs, lhs, __FILE__);
          }
        }
      }
    }
  }
}

template class SparsifiedLaplacianInterior<POLY1>;
template class SparsifiedLaplacianInterior<POLY2>;
template class SparsifiedLaplacianInterior<POLY3>;
template class SparsifiedLaplacianInterior<POLY4>;

} // namespace nalu
} // namespace Sierra

