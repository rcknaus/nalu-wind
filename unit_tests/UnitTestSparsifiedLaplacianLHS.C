#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>


#include "kernel/SparsifiedEdgeLaplacian.h"
#include "kernel/ConductionKernel.h"
#include "master_element/DirectionMacros.h"
#include "CVFEMTypeDefs.h"
#include "SimdInterface.h"
#include "KokkosInterface.h"
#include "element_promotion/NodeMapMaker.h"
#include "element_promotion/QuadratureRule.h"

#include "master_element/TensorOps.h"

#include "DoubleTypeComparisonMacros.h"

namespace sierra { namespace nalu {

template <int p> LocalArray<DoubleType[p][p][p][8][8]> compute_sparsified_laplacian_matrix(bool print = false)
{
  double Q[3][3] = {
      {0,0,-1},{0,1,-1},{0.5,0.5,1}
  };
  std::cout << "detj: " << determinant33(&Q[0][0]) << std::endl;
  ThrowRequire(determinant33(&Q[0][0]) > 0);

  nodal_vector_array<DoubleType, p> work_coords;
  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p+1; ++j) {
      for (int i = 0; i < p+1;++i) {
        double x = coords1D[i];
        double y = coords1D[j];
        double z = coords1D[k];

        double xvec[3] = {
            Q[0][0] * x + Q[0][1] * y + Q[0][2] * z,
            Q[1][0] * x + Q[1][1] * y + Q[1][2] * z,
            Q[2][0] * x + Q[2][1] * y + Q[2][2] * z
        };

        work_coords(k, j, i, 0) = xvec[0];
        work_coords(k, j, i, 1) = xvec[1];
        work_coords(k, j, i, 2) = xvec[2];
      }
    }
  }
  auto elem_coords = la::make_view(work_coords);
  auto lhs = sparsified_laplacian_lhs<p>(elem_coords);

  if (print) {
    for (int n = 0; n < p; ++n) {
      for (int m = 0; m < p; ++m) {
        for (int l = 0; l < p; ++l) {
          for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
              std::cout << lhs(0, 0, 0, j, i)[0] << ", ";
            }
            std::cout << std::endl;
          }
          std::cout << "-----(" << l + m * p + n * p * p << ")" << std::endl;
        }
      }
    }
  }

  return lhs;
}


TEST(SparsifiedLaplacian, p1_is_edge)
{
  auto lhs = compute_sparsified_laplacian_matrix<1>(true);
  static constexpr LocalArray<double[8][8]> edge_lhs = {{
    { 1.5, -0.5, 0, -0.5, -0.5, 0, 0, 0,  },
    { -0.5, 1.5, -0.5, 0, 0, -0.5, 0, 0,  },
    { 0, -0.5, 1.5, -0.5, 0, 0, -0.5, 0,  },
    { -0.5, 0, -0.5, 1.5, 0, 0, 0, -0.5,  },
    { -0.5, 0, 0, 0, 1.5, -0.5, 0, -0.5,  },
    { 0, -0.5, 0, 0, -0.5, 1.5, -0.5, 0,  },
    { 0, 0, -0.5, 0, 0, -0.5, 1.5, -0.5,  },
    { 0, 0, 0, -0.5, -0.5, 0, -0.5, 1.5,  },
  }};

  static constexpr int perm[8] = {0,1,3,2,4,5,7,6};
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      EXPECT_DOUBLE_EQ(edge_lhs(j, i), lhs(0, 0, 0, perm[j], perm[i])[0]);
    }
  }
}

namespace {
  LocalArray<DoubleType[2][2][2][3]> prism_coordinates(Kokkos::Array<double, 3> delta, Kokkos::Array<double, 3> lowerLeft = {{0,0,0}})
  {
    auto box = LocalArray<DoubleType[2][2][2][3]>();
    box(0,0,0,XH) = lowerLeft[XH];
    box(0,0,1,XH) = lowerLeft[XH] + delta[XH];
    box(0,1,0,XH) = lowerLeft[XH];
    box(0,1,1,XH) = lowerLeft[XH] + delta[XH];
    box(1,0,0,XH) = lowerLeft[XH];
    box(1,0,1,XH) = lowerLeft[XH] + delta[XH];
    box(1,1,0,XH) = lowerLeft[XH];
    box(1,1,1,XH) = lowerLeft[XH] + delta[XH];

    box(0,0,0,YH) = lowerLeft[YH];
    box(0,0,1,YH) = lowerLeft[YH];
    box(0,1,0,YH) = lowerLeft[YH] + delta[YH];
    box(0,1,1,YH) = lowerLeft[YH] + delta[YH];
    box(1,0,0,YH) = lowerLeft[YH];
    box(1,0,1,YH) = lowerLeft[YH];
    box(1,1,0,YH) = lowerLeft[YH] + delta[YH];
    box(1,1,1,YH) = lowerLeft[YH] + delta[YH];

    box(0,0,0,ZH) = lowerLeft[ZH];
    box(0,0,1,ZH) = lowerLeft[ZH];
    box(0,1,0,ZH) = lowerLeft[ZH];
    box(0,1,1,ZH) = lowerLeft[ZH];
    box(1,0,0,ZH) = lowerLeft[ZH] + delta[ZH];
    box(1,0,1,ZH) = lowerLeft[ZH] + delta[ZH];
    box(1,1,0,ZH) = lowerLeft[ZH] + delta[ZH];
    box(1,1,1,ZH) = lowerLeft[ZH] + delta[ZH];

//
//    (-3.14159, -3.14159, 3.14159),
//    (-3.14159, -3.14159, 2.0944),
//    (-1.0472, -2.0944, 2.0944),
//    (-3.14159, -2.0944, 3.14159),
//    (-2.0944, -3.14159, 3.14159),
//    (-2.0944, -1.0472, 2.0944),
//    (-3.14159, -2.0944, 2.0944),
//    (-2.0944, -2.0944, 1.0472)

    return box;
  }
}

TEST(SparsifiedLaplacian, p2_has_one_sort)
{
  auto lhs = compute_sparsified_laplacian_matrix<2>();

  auto sub_element_box = prism_coordinates({{1,1,1}});
  auto box = la::make_view(sub_element_box);
  auto individual_lhs = sparsified_laplacian_lhs<1>(box);
  for (int n = 0; n < 2; ++n) {
    for (int m = 0; m < 2; ++m) {
      for (int l = 0; l < 2; ++l) {
        for (int j = 0; j < 8; ++j) {
          for (int i = 0; i < 8; ++i) {
            std::cout << lhs(n, m, l, j, i)[0] << " | ";

//            EXPECT_DOUBLETYPE_EQ(individual_lhs(0, 0, 0, j, i), lhs(n, m, l, j, i));
          }
          std::cout << std::endl;
        }
        std::cout << "----------" << std::endl;
      }
    }
  }
}

TEST(SparsifiedLaplacian, p3_check_all)
{
  constexpr int p = 3;
  auto coords1D = gauss_lobatto_legendre_rule(p+1).first;
  const auto deltas = std::array<double, p>{
    coords1D[1] - coords1D[0], coords1D[2] - coords1D[1], coords1D[3] - coords1D[2],
  };
  auto laplacians = Kokkos::View<LocalArray<DoubleType[1][1][1][8][8]>[p][p][p]>("boxes");
  for (int n = 0; n < p; ++n) {
    for (int m = 0; m < p; ++m) {
      for (int l = 0; l < p; ++l) {
        auto box = prism_coordinates({{deltas[l], deltas[m], deltas[n]}});
        auto box_view = la::make_view(box);
        laplacians(n,m,l) = sparsified_laplacian_lhs<1>(box_view);
      }
    }
  }

  auto lhs = compute_sparsified_laplacian_matrix<p>();
  for (int n = 0; n < p; ++n) {
    for (int m = 0; m < p; ++m) {
      for (int l = 0; l < p; ++l) {
        for (int j = 0; j < 8; ++j) {
          for (int i = 0; i < 8; ++i) {
            EXPECT_DOUBLE_EQ(laplacians(n, m, l)(0, 0, 0, j, i)[0], lhs(n, m, l, j, i)[0]);
          }
        }
      }
    }
  }

}

}}
