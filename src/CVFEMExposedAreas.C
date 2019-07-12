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

//namespace {
//
template <int p> face_view::vector_view<p> exposed_area_vectors(face_view::vector_view<p> coordinates)
{
  auto ops = CVFEMOperators<p, DoubleType>();
  auto area_vecs = face_view::vector_view<p>("area_v" + std::to_string(rand()), coordinates.extent_int(0));
  for (int index  = 0; index < coordinates.extent_int(0); ++index) {
    auto local_coords = face_nodal_vector_view<DoubleType, p>(&coordinates(index,0,0,0,0));
    auto work_metric = la::zero<face_nodal_vector_array<DoubleType, p>>();
    auto metric = la::make_view(work_metric);
    high_order_metrics::compute_exposed_area_linear(ops, local_coords, metric);

    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        area_vecs(index, j, i, XH) = metric(j, i, XH);
        area_vecs(index, j, i, YH) = metric(j, i, YH);
        area_vecs(index, j, i, ZH) = metric(j, i, ZH);
      }
    }
  }
  return area_vecs;
}


template face_view::vector_view<POLY1> exposed_area_vectors<POLY1>(face_view::vector_view<POLY1>);
template face_view::vector_view<POLY2> exposed_area_vectors<POLY2>(face_view::vector_view<POLY2>);
template face_view::vector_view<POLY3> exposed_area_vectors<POLY3>(face_view::vector_view<POLY3>);
template face_view::vector_view<POLY4> exposed_area_vectors<POLY4>(face_view::vector_view<POLY4>);

}}
