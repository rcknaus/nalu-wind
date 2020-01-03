#include <math.h>
#include <stdlib.h>

#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <algorithm>
#include <random>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/FieldState.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionJacobiPreconditioner.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "StkConductionFixture.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/make_rcp.hpp"
#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

class JacobiFixture : public ConductionFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int nx = 3;
  static constexpr double scale = nx;

  JacobiFixture()
    : ConductionFixture(nx, scale),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part())),
      elid(entity_to_row_lid_mapping(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      offsets(create_offset_map<order>(mesh, meta.universal_part(), elid)),
      owned_map(owned_row_map(mesh, gid_field, meta.universal_part())),
      shared_map(owned_and_shared_row_map(
        mesh, gid_field, tpetra_gid_field, meta.universal_part())),
      exporter(shared_map, owned_map),
      coordinate_ordinal(coordinate_field().mesh_meta_data_ordinal()),
      residual_ordinals(conduction_field_ordinals(meta)),
      coefficient_ordinals(conduction_coefficient_ordinals(meta))
  {
  }

  elem_mesh_index_view<order> conn;
  entity_row_view_type elid;
  elem_offset_view<order> offsets;

  Teuchos::RCP<const Tpetra::Map<>> owned_map;
  Teuchos::RCP<const Tpetra::Map<>> shared_map;
  Tpetra::Export<> exporter;

  int coordinate_ordinal{-1};
  Kokkos::Array<int, conduction_info::num_physics_fields> residual_ordinals;
  Kokkos::Array<int, conduction_info::num_coefficient_fields>
    coefficient_ordinals;
};

TEST_F(JacobiFixture, jacobi_operator_is_stricly_positive_for_laplacian)
{
  auto fields = gather_required_conduction_fields<order>(
    conn, fm, coordinate_ordinal, residual_ordinals);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  JacobiOperator<order> jac_op(offsets, exporter);
  jac_op.set_coefficients(0.0, coefficient_fields);
  jac_op.compute_diagonal();
  auto& result = jac_op.get_inverse_diagonal();
  result.sync_host();
  auto view_h = result.getLocalViewHost();
  for (size_t k = 0u; k < result.getLocalLength(); ++k) {
    ASSERT_GT(view_h(k, 0), 1.0e-2);
  }
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
