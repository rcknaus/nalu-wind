#include "CVFEMAreas.h"

#include "CVFEMTypeDefs.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "MatrixFreeTraits.h"

#include "master_element/TensorProductCVFEMAdvectionMetric.h"
#include "element_promotion/NodeMapMaker.h"

#include "stk_util/util/ReportHandler.hpp"

namespace sierra { namespace nalu {

template <int p> ko::scs_vector_view<p> areas(ko::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto areav = ko::scs_vector_view<p>("area_vectors", coordinates.extent_int(0));
  for (int index = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index, 0, 0, 0, 0));
    auto work_metric = la::zero<scs_vector_array<DoubleType, p>>();
    auto elem_areas = la::make_view(work_metric);
    high_order_metrics::compute_area_linear(ops, local_coords, elem_areas);

    for (int dj = 0; dj < 3; ++dj) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int di = 0; di < 3; ++di) {
              areav(index, dj, k, j, i, di) = elem_areas(dj, k, j, i, di);
            }
          }
        }
      }
    }
  }
  return areav;
}
template ko::scs_vector_view<POLY1> areas<POLY1>(ko::vector_view<POLY1>);
template ko::scs_vector_view<POLY2> areas<POLY2>(ko::vector_view<POLY2>);
template ko::scs_vector_view<POLY3> areas<POLY3>(ko::vector_view<POLY3>);
template ko::scs_vector_view<POLY4> areas<POLY4>(ko::vector_view<POLY4>);

}}
