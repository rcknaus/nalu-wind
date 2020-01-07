#include "matrix_free/ConductionFields.h"

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_ngp/NgpFieldManager.hpp"
#include "stk_topology/topology.hpp"

#include <stdexcept>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {
int
stk_ordinal(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState fs = stk::mesh::StateNone)
{
  ThrowRequireMsg(
    meta.get_field(stk::topology::NODE_RANK, name), conduction_info::q_name);

  ThrowRequireMsg(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(fs),
    std::string(conduction_info::q_name) + " has no state " +
      std::to_string((fs)));

  return static_cast<int>(meta.get_field(stk::topology::NODE_RANK, name)
                            ->field_state(fs)
                            ->mesh_meta_data_ordinal());
}
} // namespace

Kokkos::Array<int, conduction_info::num_physics_fields>
conduction_field_ordinals(const stk::mesh::MetaData& meta)
{
  Kokkos::Array<int, conduction_info::num_physics_fields> cf;
  cf[conduction_info::TEMPERATURE_NP1] =
    stk_ordinal(meta, conduction_info::q_name, stk::mesh::StateNP1);
  cf[conduction_info::TEMPERATURE_NP0] =
    stk_ordinal(meta, conduction_info::q_name, stk::mesh::StateN);

  ThrowRequireMsg(
    meta.get_field(stk::topology::NODE_RANK, conduction_info::q_name)
      ->field_state(stk::mesh::StateNM1),
    "Ony BDF2 is supported with matrix free");

  cf[conduction_info::TEMPERATURE_NM1] =
    stk_ordinal(meta, conduction_info::q_name, stk::mesh::StateNM1);
  cf[conduction_info::ALPHA] =
    stk_ordinal(meta, conduction_info::volume_weight_name);
  cf[conduction_info::LAMBDA] =
    stk_ordinal(meta, conduction_info::diffusion_weight_name);
  return cf;
}

Kokkos::Array<int, conduction_info::num_coefficient_fields>
conduction_coefficient_ordinals(const stk::mesh::MetaData& meta)
{
  return Kokkos::Array<int, conduction_info::num_coefficient_fields>{{
    stk_ordinal(meta, conduction_info::volume_weight_name),
    stk_ordinal(meta, conduction_info::diffusion_weight_name),
  }};
}

int
dirichlet_bc_ordinal(const stk::mesh::MetaData& meta)
{
  if (
    meta.get_field(stk::topology::NODE_RANK, conduction_info::qbc_name) ==
    nullptr) {
    return -1;
  }
  return stk_ordinal(meta, conduction_info::qbc_name);
}

int
flux_bc_ordinal(const stk::mesh::MetaData& meta)
{
  if (
    meta.get_field(stk::topology::NODE_RANK, conduction_info::flux_name) ==
    nullptr) {
    return -1;
  }
  return stk_ordinal(meta, conduction_info::flux_name);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

template <int p>
ResidualFields<p>
gather_required_conduction_fields_t<p>::invoke(
  const_elem_mesh_index_view<p> conn,
  const ngp::FieldManager& fm,
  int coordinate_ordinal,
  Kokkos::Array<int, conduction_info::num_physics_fields> ordinals)
{
  ResidualFields<p> fields;

  fields.coordinate_ordinal = coordinate_ordinal;
  fields.ordinals = ordinals;

  fields.qp1 = scalar_view<p>{"qp1", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, fm.get_field<double>(ordinals[conduction_info::TEMPERATURE_NP1]),
    fields.qp1);
  fields.qp0 = scalar_view<p>{"qp0", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, fm.get_field<double>(ordinals[conduction_info::TEMPERATURE_NP0]),
    fields.qp0);
  fields.qm1 = scalar_view<p>{"qm1", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, fm.get_field<double>(ordinals[conduction_info::TEMPERATURE_NM1]),
    fields.qm1);

  vector_view<p> coords{"coords", conn.extent(0)};
  stk_simd_vector_field_gather<p>(
    conn, fm.get_field<double>(coordinate_ordinal), coords);

  scalar_view<p> alpha{"alpha", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, fm.get_field<double>(ordinals[conduction_info::ALPHA]), alpha);
  fields.volume_metric = geom::volume_metric<p>(alpha, coords);

  scalar_view<p> lambda{"lambda", conn.extent(0)};
  stk_simd_scalar_field_gather<p>(
    conn, fm.get_field<double>(ordinals[conduction_info::LAMBDA]), lambda);
  fields.diffusion_metric = geom::diffusion_metric<p>(lambda, coords);

  return fields;
}
INSTANTIATE_POLYSTRUCT(gather_required_conduction_fields_t);

} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
