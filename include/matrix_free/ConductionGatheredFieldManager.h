#ifndef FIELD_MANAGER_H
#define FIELD_MANAGER_H

#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosFramework.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ConductionGatheredFieldManager
{
public:
  ConductionGatheredFieldManager(
    const stk::mesh::MetaData&,
    const ngp::Mesh&,
    const ngp::FieldManager&,
    stk::mesh::Selector,
    stk::mesh::Selector = {},
    stk::mesh::Selector = {});

  void gather_all();
  void update_solution_fields();
  void swap_states();

  ResidualFields<p> get_residual_fields() { return fields_; }
  BCFields get_bc_fields() { return bc_fields_; }
  LinearizedResidualFields<p> get_coefficient_fields()
  {
    return coefficient_fields_;
  }
  BCFluxFields<p> get_flux_fields() { return flux_fields_; }

private:
  const ngp::Mesh& mesh;
  const ngp::FieldManager& fm;

  const stk::mesh::Selector active_;
  const const_elem_mesh_index_view<p> conn_;
  const Kokkos::Array<int, conduction_info::num_physics_fields>
    conduction_ordinals_;
  ResidualFields<p> fields_;
  LinearizedResidualFields<p> coefficient_fields_;

  const stk::mesh::Selector dirichlet_;
  const const_node_mesh_index_view dirichlet_nodes_;
  const int dirichlet_ordinal_{-1};
  BCFields bc_fields_;

  const stk::mesh::Selector flux_;
  const const_face_mesh_index_view<p> flux_faces_;
  const int flux_ordinal_{-1};
  BCFluxFields<p> flux_fields_;

  const int coordinate_ordinal_{-1};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
