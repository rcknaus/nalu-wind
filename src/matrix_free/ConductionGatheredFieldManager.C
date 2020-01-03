#include "matrix_free/ConductionGatheredFieldManager.h"

#include <stk_mesh/base/Types.hpp>

#include "matrix_free/ConductionFields.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/MetaData.hpp"

#include "stk_ngp/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionGatheredFieldManager<p>::ConductionGatheredFieldManager(
  const stk::mesh::MetaData& meta,
  const ngp::Mesh& mesh_in,
  const ngp::FieldManager& fm_in,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  stk::mesh::Selector flux_in)
  : mesh(mesh_in),
    fm(fm_in),
    active_(active_in),
    conn_(stk_connectivity_map<p>(mesh, active_)),
    conduction_ordinals_(conduction_field_ordinals(meta)),
    dirichlet_(dirichlet_in),
    dirichlet_nodes_(simd_node_map(mesh, dirichlet_)),
    dirichlet_ordinal_(dirichlet_bc_ordinal(meta)),
    flux_(flux_in),
    flux_faces_(face_node_map<p>(mesh, flux_in)),
    flux_ordinal_(flux_bc_ordinal(meta)),
    coordinate_ordinal_(meta.coordinate_field()->mesh_meta_data_ordinal())
{
}

template <int p>
void
ConductionGatheredFieldManager<p>::gather_all()
{
  ngp::ProfilingBlock pf("ConductionGatheredFieldManager<p>::gather_all");

  fields_ = gather_required_conduction_fields<p>(
    conn_, fm, coordinate_ordinal_, conduction_ordinals_);
  coefficient_fields_.volume_metric = fields_.volume_metric;
  coefficient_fields_.diffusion_metric = fields_.diffusion_metric;

  if (dirichlet_nodes_.extent_int(0) > 0) {
    bc_fields_.qp1 =
      node_scalar_view("qp1_at_bc", dirichlet_nodes_.extent_int(0));
    stk_simd_scalar_node_gather(
      dirichlet_nodes_,
      fm.get_field<double>(
        conduction_ordinals_[conduction_info::TEMPERATURE_NP1]),
      bc_fields_.qp1);

    bc_fields_.qbc =
      node_scalar_view("qspecified_at_bc", dirichlet_nodes_.extent_int(0));
    stk_simd_scalar_node_gather(
      dirichlet_nodes_, fm.get_field<double>(dirichlet_ordinal_),
      bc_fields_.qbc);
  }

  if (flux_faces_.extent_int(0) > 0) {
    {
      auto face_coords =
        face_vector_view<p>("face_coords", flux_faces_.extent_int(0));
      stk_simd_face_vector_field_gather<p>(
        flux_faces_, fm.get_field<double>(coordinate_ordinal_), face_coords);
      flux_fields_.exposed_areas = geom::exposed_areas<p>(face_coords);
    }
    flux_fields_.flux = face_scalar_view<p>("flux", flux_faces_.extent_int(0));
    stk_simd_face_scalar_field_gather<p>(
      flux_faces_, fm.get_field<double>(flux_ordinal_), flux_fields_.flux);
  }
}

template <int p>
void
ConductionGatheredFieldManager<p>::update_solution_fields()
{
  ngp::ProfilingBlock pf(
    "ConductionGatheredFieldManager<p>::update_solution_fields");

  stk_simd_scalar_field_gather<p>(
    conn_,
    fm.get_field<double>(
      conduction_ordinals_[conduction_info::TEMPERATURE_NP1]),
    fields_.qp1);
  if (dirichlet_nodes_.extent_int(0) > 0) {
    stk_simd_scalar_node_gather(
      dirichlet_nodes_,
      fm.get_field<double>(
        conduction_ordinals_[conduction_info::TEMPERATURE_NP1]),
      bc_fields_.qp1);
  }
}

template <int p>
void
ConductionGatheredFieldManager<p>::swap_states()
{
  ngp::ProfilingBlock pf("ConductionGatheredFieldManager<p>::swap_states");

  auto qm1 = fields_.qm1;
  fields_.qm1 = fields_.qp0;
  fields_.qp0 = fields_.qp1;
  fields_.qp1 = qm1;
}
INSTANTIATE_POLYCLASS(ConductionGatheredFieldManager);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
