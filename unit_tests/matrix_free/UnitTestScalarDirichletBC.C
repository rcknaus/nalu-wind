#include "matrix_free/ScalarDirichletBC.h"
#include "matrix_free/StkEntityToRowMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/ConductionFields.h"

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

class DirichletFixture : public ConductionFixture
{
protected:
  DirichletFixture()
    : ConductionFixture(nx, scale),
      stk_entity_to_tpetra_index(entity_to_row_lid_mapping(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      dirichlet_nodes(simd_node_map(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4))),
      dirichlet_offsets(simd_node_offsets(
        mesh,
        meta.get_topology_root_part(stk::topology::QUAD_4),
        stk_entity_to_tpetra_index)),
      owned_map(owned_row_map(mesh, gid_field, meta.universal_part())),
      shared_map(owned_and_shared_row_map(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      exporter(shared_map, owned_map),
      coordinate_ordinal(coordinate_field().mesh_meta_data_ordinal()),
      bc_ordinal(dirichlet_bc_ordinal(meta)),
      residual_ordinals(conduction_field_ordinals(meta)),
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
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) = 5.0;
        *stk::mesh::field_data(qbc_field, node) = -2.3;
      }
    }
  }
  static constexpr int nx = 4;
  static constexpr double scale = M_PI;
  const const_entity_row_view_type stk_entity_to_tpetra_index;
  const const_node_mesh_index_view dirichlet_nodes;
  const const_node_offset_view dirichlet_offsets;
  Teuchos::RCP<const Tpetra::Map<>> owned_map;
  Teuchos::RCP<const Tpetra::Map<>> shared_map;
  Tpetra::Export<> exporter;

  int coordinate_ordinal{-1};
  int bc_ordinal{-1};
  Kokkos::Array<int, conduction_info::num_physics_fields> residual_ordinals;

  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;

  Tpetra::MultiVector<> shared_lhs;
  Tpetra::MultiVector<> shared_rhs;
};

TEST_F(DirichletFixture, bc_residual)
{
  auto qp1 = node_scalar_view("qp1_at_bc", dirichlet_nodes.extent_int(0));
  stk_simd_scalar_node_gather(
    dirichlet_nodes,
    fm.get_field<double>(residual_ordinals[conduction_info::TEMPERATURE_NP1]),
    qp1);

  auto qbc =
    node_scalar_view("qspecified_at_bc", dirichlet_nodes.extent_int(0));
  stk_simd_scalar_node_gather(
    dirichlet_nodes, fm.get_field<double>(bc_ordinal), qbc);

  shared_rhs.putScalar(0.);
  scalar_dirichlet_residual(
    dirichlet_offsets, qp1, qbc, owned_rhs.getLocalLength(),
    shared_rhs.getLocalViewDevice());
  shared_rhs.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(shared_rhs, exporter, Tpetra::ADD);

  owned_rhs.sync_host();
  auto view_h = owned_rhs.getLocalViewHost();

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    maxval = std::max(maxval, std::abs(view_h(k, 0)));
  }
  ASSERT_DOUBLE_EQ(maxval, 7.3);
}

TEST_F(DirichletFixture, linearized_bc_residual)
{
  constexpr double some_val = 85432.2;
  owned_lhs.putScalar(some_val);

  shared_lhs.doImport(owned_lhs, exporter, Tpetra::INSERT);

  scalar_dirichlet_linearized(
    dirichlet_offsets, owned_lhs.getLocalLength(),
    shared_lhs.getLocalViewDevice(), shared_rhs.getLocalViewDevice());

  shared_rhs.modify_device();
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(shared_rhs, exporter, Tpetra::ADD);

  owned_rhs.sync_host();
  auto view_h = owned_rhs.getLocalViewHost();

  constexpr double tol = 1.0e-14;
  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    const bool zero_or_val = std::abs(view_h(k, 0) - 0) < tol ||
                             std::abs(view_h(k, 0) - some_val) < tol;
    ASSERT_TRUE(zero_or_val);
    maxval = std::max(maxval, view_h(k, 0));
  }
  ASSERT_DOUBLE_EQ(maxval, some_val);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
