#ifndef CONDUCTION_FIELDS_H
#define CONDUCTION_FIELDS_H

#include "matrix_free/ConductionInfo.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

#include "Kokkos_Array.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/Types.hpp"
#include "stk_ngp/NgpFieldManager.hpp"

namespace stk {
namespace mesh {
class MetaData;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

struct BCFields
{
  node_scalar_view qp1;
  node_scalar_view qbc;
};

template <int p>
struct BCFluxFields
{
  face_scalar_view<p> flux;
  face_vector_view<p> exposed_areas;
};

template <int p>
struct ResidualFields
{
  static constexpr int num_fields = conduction_info::num_physics_fields;
  scalar_view<p> qm1;
  scalar_view<p> qp0;
  scalar_view<p> qp1;
  scalar_view<p> volume_metric;
  scs_vector_view<p> diffusion_metric;

  int coordinate_ordinal{-1};
  Kokkos::Array<int, num_fields> ordinals;
};

Kokkos::Array<int, conduction_info::num_physics_fields>
conduction_field_ordinals(const stk::mesh::MetaData&);
Kokkos::Array<int, conduction_info::num_coefficient_fields>
conduction_coefficient_ordinals(const stk::mesh::MetaData&);
int dirichlet_bc_ordinal(const stk::mesh::MetaData&);
int flux_bc_ordinal(const stk::mesh::MetaData&);

namespace impl {

template <int p>
struct gather_required_conduction_fields_t
{
  static ResidualFields<p> invoke(
    const_elem_mesh_index_view<p>,
    const ngp::FieldManager&,
    int,
    Kokkos::Array<int, conduction_info::num_physics_fields>);
};

} // namespace impl
P_INVOKEABLE(gather_required_conduction_fields)

template <int p>
struct LinearizedResidualFields
{
  static constexpr int num_fields = conduction_info::num_coefficient_fields;
  scalar_view<p> volume_metric;
  scs_vector_view<p> diffusion_metric;

  int coordinate_ordinal{-1};
  Kokkos::Array<int, num_fields> ordinals;
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
