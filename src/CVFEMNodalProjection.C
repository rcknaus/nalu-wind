/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "CVFEMNodalProjection.h"

#include "MatrixFreeTraits.h"
#include "SimdFieldGather.h"
#include "master_element/TensorProductCVFEMVolumeMetric.h"
#include "ExecutePolyFunction.h"
#include "CVFEMVolumes.h"

#include <CVFEMTypeDefs.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <stk_util/util/ReportHandler.hpp>
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"


namespace sierra { namespace nalu {

template <int p> void nodal_average(
  elem_entity_view_t<p> elemNodes,
  elem_view::scalar_view<p> local_vol,
  elem_view::scalar_view<p> dnv,
  elem_view::scalar_view<p> q,
  ScalarFieldType& qField)
{
  const auto ops = CVFEMOperators<p>();
  for (int index = 0; index < elemNodes.extent_int(0); ++index) {
    auto work_vol_weighted_field = nodal_scalar_array<DoubleType, p>();
    static constexpr int n1D = p + 1;
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          work_vol_weighted_field(k,j,i) = q(index, k, j, i) * local_vol(index, k, j, i);
        }
      }
    }

    auto vol_weighted_field = la::make_view(work_vol_weighted_field);
    auto work_integrated_field = la::zero<nodal_scalar_array<DoubleType, p>>();
    auto integrated_field = la::make_view(work_integrated_field);
    ops.volume(vol_weighted_field, integrated_field);

    for (int n = 0; n < simdLen && elemNodes(index, n, 0,0,0).is_local_offset_valid(); ++n) {
      for (int k = 0; k < n1D; ++k) {
        for (int j = 0; j < n1D; ++j) {
          for (int i = 0; i < n1D; ++i) {
            const auto node = elemNodes(index, n, k, j, i);
            *stk::mesh::field_data(qField, node) +=
                stk::simd::get_data(integrated_field(k, j, i) / dnv(index, k,j,i), n);
          }
        }
      }
    }
  }
}

template void nodal_average<POLY1>(
  elem_entity_view_t<POLY1> elemNodes,
  elem_view::scalar_view<POLY1> local_vol,
  elem_view::scalar_view<POLY1> dnv,
  elem_view::scalar_view<POLY1> q,
  ScalarFieldType& qField);

template void nodal_average<POLY2>(
  elem_entity_view_t<POLY2> elemNodes,
  elem_view::scalar_view<POLY2> local_vol,
  elem_view::scalar_view<POLY2> dnv,
  elem_view::scalar_view<POLY2> q,
  ScalarFieldType& qField);

template void nodal_average<POLY3>(
  elem_entity_view_t<POLY3> elemNodes,
  elem_view::scalar_view<POLY3> local_vol,
  elem_view::scalar_view<POLY3> dnv,
  elem_view::scalar_view<POLY3> q,
  ScalarFieldType& qField);

template void nodal_average<POLY4>(
  elem_entity_view_t<POLY4> elemNodes,
  elem_view::scalar_view<POLY4> local_vol,
  elem_view::scalar_view<POLY4> dnv,
  elem_view::scalar_view<POLY4> q,
  ScalarFieldType& qField);

namespace {

template <int p>
void nodal_average_field(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  ScalarFieldType& qField)
{
  const auto& dnvField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  auto entities = element_entity_view<p>(bulk, selector);
  auto vol = volumes<p>(gather_field<p>(bulk, selector, coordField));
  auto dnv = gather_field<p>(bulk, selector, dnvField);
  auto q = gather_field<p>(bulk, selector, qField);

  stk::mesh::field_fill(0.0, qField);
  nodal_average<p>(entities, vol, dnv, q, qField);
  stk::mesh::parallel_sum(bulk, {&qField});
}
MAKE_INVOKEABLE_P(nodal_average_field);

}

void nodal_average_field(
  int p,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  ScalarFieldType& qField)
{
  execute_poly_function<nodal_average_field_invokeable>(p, bulk, selector, coordField, qField);
}


namespace{

template <int p, typename Scalar>
nodal_tensor_array<Scalar, p> nodal_grad_u(
  const CVFEMOperators<p, Scalar>& ops,
  const nodal_vector_view<p, Scalar>& xc,
  const nodal_vector_view<p, Scalar>& q)
{
  NALU_ALIGNED Scalar base_box[3][8];
  hex_vertex_coordinates<p, Scalar>(xc, base_box);

  nodal_tensor_array<Scalar, p> work_gradq;
  auto nodal_gradq = la::make_view(work_gradq);

  nodal_tensor_array<Scalar, p> phys_gradq;

  const auto& nodalInterp = ops.mat_.linearNodalInterp;

  ops.nodal_grad(q, nodal_gradq);
  static constexpr int n1D = p + 1;
  for (int k = 0; k < n1D; ++k) {
    NALU_ALIGNED const Scalar interpk[2] = { nodalInterp(0, k), nodalInterp(1, k) };
    for (int j = 0; j < n1D; ++j) {
      NALU_ALIGNED const Scalar interpj[2] = { nodalInterp(0, j), nodalInterp(1, j) };
      for (int i = 0; i < n1D; ++i) {
        NALU_ALIGNED const Scalar interpi[2] = { nodalInterp(0, i), nodalInterp(1, i) };

        NALU_ALIGNED Scalar jact[3][3];
        hex_jacobian_t(base_box, interpi, interpj, interpk, jact);

        NALU_ALIGNED Scalar invJac[3][3];
        invert_matrix33(jact, invJac);

        for (int d = 0; d < 3; ++d)  {
          phys_gradq(k, j, i, d, XH) = invJac[XH][XH] * nodal_gradq(k, j, i, d, XH)
                        + invJac[XH][YH] * nodal_gradq(k, j, i, d, YH) + invJac[XH][ZH] * nodal_gradq(k, j, i, d, ZH);

          phys_gradq(k, j, i, d, YH) = invJac[YH][XH] * nodal_gradq(k, j, i, d, XH)
                        + invJac[YH][YH] * nodal_gradq(k, j, i, d, YH) + invJac[YH][ZH] * nodal_gradq(k, j, i, d, ZH);

          phys_gradq(k, j, i, d, ZH) = invJac[ZH][XH] * nodal_gradq(k, j, i, d, XH)
                        + invJac[ZH][YH] * nodal_gradq(k, j, i, d, YH) + invJac[ZH][ZH] * nodal_gradq(k, j, i, d, ZH);
        }
      }
    }
  }
  return phys_gradq;
}

template <int p> void grad_field(
  const elem_entity_view_t<p> elemNodes,
  const elem_view::vector_view<p>& coordinates,
  const elem_view::scalar_view<p> dnv,
  const elem_view::vector_view<p> q,
  GenericFieldType& dqdxField)
{
  const auto ops = CVFEMOperators<p>();
  for (int index = 0; index < elemNodes.extent_int(0); ++index) {
    auto local_coords = nodal_vector_view<p, DoubleType>(&coordinates(index,0,0,0,0));
    auto local_q = nodal_vector_view<p,DoubleType>(&q(index,0,0,0,0));
    const auto dqdx = nodal_grad_u<p>(ops, local_coords, local_q);

    auto work_local_vol = la::zero<nodal_scalar_array<DoubleType, p>>();
    auto local_vol = la::make_view(work_local_vol);
    high_order_metrics::compute_volume_metric_linear(ops, local_coords, local_vol);

    for (int dj = 0; dj < 3; ++dj) {
      auto work_vol_weighted_field = nodal_vector_array<DoubleType, p>();
      static constexpr int n1D = p + 1;
      for (int k = 0; k < n1D; ++k) {
        for (int j = 0; j < n1D; ++j) {
          for (int i = 0; i < n1D; ++i) {
            const auto vol_val = local_vol(k, j, i);
            for (int di = 0; di < 3; ++di) {
              work_vol_weighted_field(k, j, i, di) = dqdx(k, j, i, dj, di) * vol_val;
            }
          }
        }
      }
      auto vol_weighted_field = la::make_view(work_vol_weighted_field);
      auto work_integrated_field = la::zero<nodal_vector_array<DoubleType, p>>();
      auto integrated_field = la::make_view(work_integrated_field);
      ops.volume(vol_weighted_field, integrated_field);

      for (int n = 0; n < simdLen && elemNodes(index, n, 0, 0, 0).is_local_offset_valid(); ++n) {
        for (int k = 0; k < n1D; ++k) {
          for (int j = 0; j < n1D; ++j) {
            for (int i = 0; i < n1D; ++i) {
              const auto node = elemNodes(index, n, k, j, i);
              const auto inv_dnv = 1.0 / dnv(index, k, j, i);
              auto* data_ptr = stk::mesh::field_data(dqdxField, node) + 3 * dj;
              data_ptr[XH] += stk::simd::get_data(integrated_field(k, j, i, XH) * inv_dnv, n);
              data_ptr[YH] += stk::simd::get_data(integrated_field(k, j, i, YH) * inv_dnv, n);
              data_ptr[ZH] += stk::simd::get_data(integrated_field(k, j, i, ZH) * inv_dnv, n);
            }
          }
        }
      }
    }
  }
}

template <int p>
void nodal_average_gradient(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  const VectorFieldType& qField,
  GenericFieldType& dqdxField)
{
  const auto& dnvField = *bulk.mesh_meta_data().get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  auto entities = element_entity_view<p>(bulk, selector);
  auto coordview = gather_field<p>(bulk, selector, coordField);
  auto dnv = gather_field<p>(bulk, selector, dnvField);
  auto q = gather_field<p>(bulk, selector, qField);

  grad_field<p>(entities, coordview, dnv, q, dqdxField);
  stk::mesh::parallel_sum(bulk, {&dqdxField});
}
MAKE_INVOKEABLE_P(nodal_average_gradient);

}//namespace


void nodal_average_gradient(
  int p,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  const VectorFieldType& qField,
  GenericFieldType& dqdxField)
{
  execute_poly_function<nodal_average_gradient_invokeable>(p, bulk, selector, coordField, qField, dqdxField);
}

}}
