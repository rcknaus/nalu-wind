
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

