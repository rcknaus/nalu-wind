#include "matrix_free/ScalarFluxBC.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/ConductionInfo.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosFramework.h"
#include "StkConductionFixture.h"

#include "gtest/gtest.h"

#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <algorithm>
#include <stk_simd/Simd.hpp>
#include <type_traits>

#include "mpi.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

class FluxFixture : public ConductionFixture
{
protected:
  FluxFixture()
    : ConductionFixture(nx, scale),
      stk_entity_to_tpetra_index(entity_to_row_lid_mapping(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      flux_bc_faces(face_node_map<order>(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4))),
      flux_bc_offsets(face_offsets<order>(
        mesh,
        meta.get_topology_root_part(stk::topology::QUAD_4),
        stk_entity_to_tpetra_index)),
      owned_map(owned_row_map(mesh, gid_field, meta.universal_part())),
      shared_map(owned_and_shared_row_map(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      exporter(shared_map, owned_map),
      coordinate_ordinal(coordinate_field().mesh_meta_data_ordinal()),
      flux_ordinal(flux_bc_ordinal(meta)),
      owned_lhs(owned_map, 1),
      owned_rhs(owned_map, 1),
      shared_lhs(shared_map, 1),
      shared_rhs(shared_map, 1)
  {
    owned_lhs.putScalar(0.);
    owned_rhs.putScalar(0.);

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(flux_field, node) = some_value;
      }
    }
  }

  const const_entity_row_view_type stk_entity_to_tpetra_index;
  const const_face_mesh_index_view<order> flux_bc_faces;
  const const_face_offset_view<order> flux_bc_offsets;
  Teuchos::RCP<const Tpetra::Map<>> owned_map;
  Teuchos::RCP<const Tpetra::Map<>> shared_map;
  Tpetra::Export<> exporter;

  int coordinate_ordinal{-1};
  int flux_ordinal{-1};

  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;

  Tpetra::MultiVector<> shared_lhs;
  Tpetra::MultiVector<> shared_rhs;

  static constexpr double some_value = -2.3;
  static constexpr int nx = 4;
  static constexpr double scale = 1;
};

TEST_F(FluxFixture, bc_residual)
{
  auto face_coords =
    face_vector_view<order>("face_coords", flux_bc_faces.extent_int(0));
  stk_simd_face_vector_field_gather<order>(
    flux_bc_faces, fm.get_field<double>(coordinate_ordinal), face_coords);
  auto exposed_areas = geom::exposed_areas<order>(face_coords);

  auto flux = face_scalar_view<order>("flux", flux_bc_faces.extent_int(0));
  stk_simd_face_scalar_field_gather<order>(
    flux_bc_faces, fm.get_field<double>(flux_ordinal), flux);

  shared_rhs.putScalar(0.);
  scalar_neumann_residual<order>(
    flux_bc_offsets, flux, exposed_areas, shared_rhs.getLocalViewDevice());
  shared_rhs.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(shared_rhs, exporter, Tpetra::ADD);

  owned_rhs.sync_host();
  auto view_h = owned_rhs.getLocalViewHost();

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    maxval = std::max(maxval, std::abs(view_h(k, 0)));
  }
  ASSERT_DOUBLE_EQ(maxval, std::abs(some_value / (scale * nx * scale * nx)));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
