/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include <kernel/SparsifiedEdgeLaplacian.h>

#include <SimdInterface.h>
#include <KokkosInterface.h>
#include <CVFEMTypeDefs.h>

#include <master_element/DirectionMacros.h>
#include <master_element/Hex8GeometryFunctions.h>

namespace sierra { namespace nalu {

namespace
{

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

template <int p> LocalArray<DoubleType[p][p][p][8][8]> sparsified_laplacian_lhs(const nodal_vector_view<p, DoubleType>& local_coords)
{
  auto sparsified_lhs = la::zero<LocalArray<DoubleType[p][p][p][8][8]>>();
  static constexpr double dv[2] = {-0.5, +0.5};

  for (int n = 0; n < p; ++n) {
    for (int m = 0; m < p; ++m) {
      for (int l = 0; l < p; ++l) {
        NALU_ALIGNED DoubleType box[3][8];
        hex_vertex_coordinates(n, m, l, local_coords, box);

        for (int k = 0; k < 2; ++k) {
          NALU_ALIGNED const DoubleType interpk[2] = { k == 0, k == 1 };
          for (int j = 0; j < 2; ++j) {
            NALU_ALIGNED const DoubleType interpj[2] = { j == 0, j == 1 };
            NALU_ALIGNED const DoubleType interpi[2] = { 0.5, 0.5 };

            NALU_ALIGNED DoubleType jac[3][3];
            hex_jacobian(box, interpi, interpj, interpk, jac);

            NALU_ALIGNED DoubleType adjJac[3][3];
            adjugate_matrix33(jac, adjJac);

            const DoubleType scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
//            ThrowRequire(scale_fac[0] > 0);
            const auto ajx = adjJac[XH][XH] * adjJac[XH][XH] + adjJac[XH][YH] * adjJac[XH][YH] + adjJac[XH][ZH] * adjJac[XH][ZH];
            const auto ajy = adjJac[XH][XH] * adjJac[YH][XH] + adjJac[XH][YH] * adjJac[YH][YH] + adjJac[XH][ZH] * adjJac[YH][ZH];
            const auto ajz = adjJac[XH][XH] * adjJac[ZH][XH] + adjJac[XH][YH] * adjJac[ZH][YH] + adjJac[XH][ZH] * adjJac[ZH][ZH];
            const auto dfdq = scale_fac * (ajx + dv[j] * ajy + dv[k] * ajz);

            const int rowIndex = 4 * k + 2 * j;
            const int indexL = rowIndex;
            const int indexR = rowIndex + 1;

            sparsified_lhs(n, m, l, indexL, indexL) += dfdq;
            sparsified_lhs(n, m, l, indexL, indexR) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexL) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexR) += dfdq;
          }
        }

        for (int k = 0; k < 2; ++k) {
          NALU_ALIGNED const DoubleType interpk[2] = { k == 0, k == 1 };
          for (int i = 0; i < 2; ++i) {
            NALU_ALIGNED const DoubleType interpi[2] = { i == 0, i == 1 };
            NALU_ALIGNED const DoubleType interpj[2] = { 0.5, 0.5 };

            NALU_ALIGNED DoubleType jac[3][3];
            hex_jacobian(box, interpi, interpj, interpk, jac);

            NALU_ALIGNED DoubleType adjJac[3][3];
            adjugate_matrix33(jac, adjJac);

            const DoubleType scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
            const auto ajx = adjJac[YH][XH] * adjJac[XH][XH] + adjJac[YH][YH] * adjJac[XH][YH] + adjJac[YH][ZH] * adjJac[XH][ZH];
            const auto ajy = adjJac[YH][XH] * adjJac[YH][XH] + adjJac[YH][YH] * adjJac[YH][YH] + adjJac[YH][ZH] * adjJac[YH][ZH];
            const auto ajz = adjJac[YH][XH] * adjJac[ZH][XH] + adjJac[YH][YH] * adjJac[ZH][YH] + adjJac[YH][ZH] * adjJac[ZH][ZH];
            const auto dfdq = scale_fac * (dv[i] * ajx + ajy + dv[k] * ajz);

            const int rowIndex = 4 * k + i;
            const int indexL = rowIndex;
            const int indexR = rowIndex + 2;

            sparsified_lhs(n, m, l, indexL, indexL) += dfdq;
            sparsified_lhs(n, m, l, indexL, indexR) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexL) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexR) += dfdq;
          }
        }

        for (int j = 0; j < 2; ++j) {
          NALU_ALIGNED const DoubleType interpj[2] = { j == 0, j == 1 };
          for (int i = 0; i < 2; ++i) {
            NALU_ALIGNED const DoubleType interpi[2] = { i == 0, i == 1 };
            NALU_ALIGNED const DoubleType interpk[2] = { 0.5, 0.5 };

            NALU_ALIGNED DoubleType jac[3][3];
            hex_jacobian(box, interpi, interpj, interpk, jac);

            NALU_ALIGNED DoubleType adjJac[3][3];
            adjugate_matrix33(jac, adjJac);

            const DoubleType scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
            const auto ajx = adjJac[ZH][XH] * adjJac[XH][XH] + adjJac[ZH][YH] * adjJac[XH][YH] + adjJac[ZH][ZH] * adjJac[XH][ZH];
            const auto ajy = adjJac[ZH][XH] * adjJac[YH][XH] + adjJac[ZH][YH] * adjJac[YH][YH] + adjJac[ZH][ZH] * adjJac[YH][ZH];
            const auto ajz = adjJac[ZH][XH] * adjJac[ZH][XH] + adjJac[ZH][YH] * adjJac[ZH][YH] + adjJac[ZH][ZH] * adjJac[ZH][ZH];
            const auto dfdq = scale_fac * (ajx + dv[j] * dv[j] * ajy + ajz);

            const int rowIndex = 2 * j + i;
            const int indexL = rowIndex;
            const int indexR = rowIndex + 4;
            sparsified_lhs(n, m, l, indexL, indexL) += dfdq;
            sparsified_lhs(n, m, l, indexL, indexR) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexL) -= dfdq;
            sparsified_lhs(n, m, l, indexR, indexR) += dfdq;
          }
        }
      }
    }
  }
  return sparsified_lhs;
}
template LocalArray<DoubleType[1][1][1][8][8]> sparsified_laplacian_lhs(const nodal_vector_view<1, DoubleType>&);
template LocalArray<DoubleType[2][2][2][8][8]> sparsified_laplacian_lhs(const nodal_vector_view<2, DoubleType>&);
template LocalArray<DoubleType[3][3][3][8][8]> sparsified_laplacian_lhs(const nodal_vector_view<3, DoubleType>&);
template LocalArray<DoubleType[4][4][4][8][8]> sparsified_laplacian_lhs(const nodal_vector_view<4, DoubleType>&);

template <typename ViewType, typename Scalar> KOKKOS_FORCEINLINE_FUNCTION
void hex_vertex_coordinates(int k, int j, int i, const ViewType& xc, Scalar base_box[3][8])
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

LocalArray<double[8][8]> laplacian_lhs(const double box[3][8])
{
  auto sparsified_lhs = la::zero<LocalArray<double[8][8]>>();
  static constexpr double dv[2] = {-0.5, +0.5};

//        std::cout << "coords: ";
//        for (int ss = 0; ss < 8; ++ss) {
//          std:: cout << "(" << box[0][ss][0] << ", " << box[1][ss][0] << ", " << box[2][ss][0] << "), ";
//        }
//        std::cout << std::endl;
//        exit(1);


  for (int k = 0; k < 2; ++k) {
    NALU_ALIGNED const double interpk[2] = { (k == 0) ? 1. : 0., (k == 1) ? 1. : 0.};
    for (int j = 0; j < 2; ++j) {
      NALU_ALIGNED const double interpj[2] = { (j == 0) ? 1. : 0., (j == 1) ? 1. : 0.};
      NALU_ALIGNED const double interpi[2] = { 0.5, 0.5 };

      NALU_ALIGNED double jac[3][3];
      hex_jacobian(box, interpi, interpj, interpk, jac);

      NALU_ALIGNED double adjJac[3][3];
      adjugate_matrix33(jac, adjJac);

      const double scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
      const auto ajx = adjJac[XH][XH] * adjJac[XH][XH] + adjJac[XH][YH] * adjJac[XH][YH] + adjJac[XH][ZH] * adjJac[XH][ZH];
      const auto ajy = adjJac[XH][XH] * adjJac[YH][XH] + adjJac[XH][YH] * adjJac[YH][YH] + adjJac[XH][ZH] * adjJac[YH][ZH];
      const auto ajz = adjJac[XH][XH] * adjJac[ZH][XH] + adjJac[XH][YH] * adjJac[ZH][YH] + adjJac[XH][ZH] * adjJac[ZH][ZH];
      const auto dfdq = scale_fac * (ajx + dv[j] * ajy + dv[k] * ajz);

      const int rowIndex = 4 * k + 2 * j;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 1;

      sparsified_lhs(indexL, indexL) += dfdq;
      sparsified_lhs(indexL, indexR) -= dfdq;
      sparsified_lhs(indexR, indexL) -= dfdq;
      sparsified_lhs(indexR, indexR) += dfdq;
    }
  }

  for (int k = 0; k < 2; ++k) {
    NALU_ALIGNED const double interpk[2] ={ (k == 0) ? 1. : 0., (k == 1) ? 1. : 0.};
    for (int i = 0; i < 2; ++i) {
      NALU_ALIGNED const double interpi[2] = { (i == 0) ? 1. : 0., (i == 1) ? 1. : 0.};
      NALU_ALIGNED const double interpj[2] = { 0.5, 0.5 };

      NALU_ALIGNED double jac[3][3];
      hex_jacobian(box, interpi, interpj, interpk, jac);

      NALU_ALIGNED double adjJac[3][3];
      adjugate_matrix33(jac, adjJac);

      const double scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
      const auto ajx = adjJac[YH][XH] * adjJac[XH][XH] + adjJac[YH][YH] * adjJac[XH][YH] + adjJac[YH][ZH] * adjJac[XH][ZH];
      const auto ajy = adjJac[YH][XH] * adjJac[YH][XH] + adjJac[YH][YH] * adjJac[YH][YH] + adjJac[YH][ZH] * adjJac[YH][ZH];
      const auto ajz = adjJac[YH][XH] * adjJac[ZH][XH] + adjJac[YH][YH] * adjJac[ZH][YH] + adjJac[YH][ZH] * adjJac[ZH][ZH];
      const auto dfdq = scale_fac * (dv[i] * ajx + ajy + dv[k] * ajz);

      const int rowIndex = 4 * k + i;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 2;

      sparsified_lhs(indexL, indexL) += dfdq;
      sparsified_lhs(indexL, indexR) -= dfdq;
      sparsified_lhs(indexR, indexL) -= dfdq;
      sparsified_lhs(indexR, indexR) += dfdq;
    }
  }

  for (int j = 0; j < 2; ++j) {
    NALU_ALIGNED const double interpj[2] = { (j == 0) ? 1. : 0., (j == 1) ? 1. : 0.};
    for (int i = 0; i < 2; ++i) {
      NALU_ALIGNED const double interpi[2] = { (i == 0) ? 1. : 0., (i == 1) ? 1. : 0.};
      NALU_ALIGNED const double interpk[2] = { 0.5, 0.5 };

      NALU_ALIGNED double jac[3][3];
      hex_jacobian(box, interpi, interpj, interpk, jac);

      NALU_ALIGNED double adjJac[3][3];
      adjugate_matrix33(jac, adjJac);

      const double scale_fac = 0.5 / (jac[0][0] * adjJac[0][0] + jac[1][0] * adjJac[1][0] + jac[2][0] * adjJac[2][0]);
      const auto ajx = adjJac[ZH][XH] * adjJac[XH][XH] + adjJac[ZH][YH] * adjJac[XH][YH] + adjJac[ZH][ZH] * adjJac[XH][ZH];
      const auto ajy = adjJac[ZH][XH] * adjJac[YH][XH] + adjJac[ZH][YH] * adjJac[YH][YH] + adjJac[ZH][ZH] * adjJac[YH][ZH];
      const auto ajz = adjJac[ZH][XH] * adjJac[ZH][XH] + adjJac[ZH][YH] * adjJac[ZH][YH] + adjJac[ZH][ZH] * adjJac[ZH][ZH];
      const auto dfdq = scale_fac * (ajx + dv[j] * dv[j] * ajy + ajz);

      const int rowIndex = 2 * j + i;
      const int indexL = rowIndex;
      const int indexR = rowIndex + 4;
      sparsified_lhs(indexL, indexL) += dfdq;
      sparsified_lhs(indexL, indexR) -= dfdq;
      sparsified_lhs(indexR, indexL) -= dfdq;
      sparsified_lhs(indexR, indexR) += dfdq;
    }
  }
  return sparsified_lhs;
}


} // namespace nalu
} // namespace Sierra

