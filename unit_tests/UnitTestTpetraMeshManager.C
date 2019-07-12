#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <tpetra_linsys/TpetraMeshManager.h>
#include <tpetra_linsys/SolutionPointCategorizer.h>
#include <kernel/ScalarDiffHOElemKernel.h>
#include <FieldTypeDef.h>
#include <Teuchos_ArrayRCP.hpp>
#include <nalu_make_unique.h>


#include <TimeIntegrator.h>
#include <SolutionOptions.h>

#include "MatrixFreeTypes.h"

#include <Tpetra_Operator.hpp>
#include <Tpetra_MultiVector.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosSolverManager.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosTFQMRSolMgr.hpp>
#include <BelosLSQRSolMgr.hpp>
#include <BelosStatusTestGenResNorm.hpp>

#include "UnitTestUtils.h"

namespace sierra { namespace nalu {

class TpetraMeshManagerFixture : public ::testing::Test
{
protected:
  TpetraMeshManagerFixture()
    : comm(MPI_COMM_WORLD),
      meta(3), bulk(meta, comm),
      gidField(meta.declare_field<GlobalIdFieldType>(stk::topology::NODE_RANK, "global_id_field")),
      lidField(meta.declare_field<stk::mesh::Field<int32_t>>(stk::topology::NODE_RANK, "local_id_field")),
      qField(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "q", 2)),
      rhoField(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "density", 2)),
      diffusivityField(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "diffusivity")),
      specificHeatField(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "specific_heat"))
  {
    double oneD = 1.0;
    int32_t one = 1;
    stk::mesh::EntityId one64 = 1;
    stk::mesh::put_field_on_mesh(lidField, meta.universal_part(), 1, &one);
    stk::mesh::put_field_on_mesh(gidField, meta.universal_part(), 3, &one64);
    stk::mesh::put_field_on_mesh(qField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(rhoField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(diffusivityField, meta.universal_part(), 1, &oneD);
    stk::mesh::put_field_on_mesh(specificHeatField, meta.universal_part(), 1, &oneD);
    unit_test_utils::fill_hex8_mesh(meshSpec, bulk);

    auto& coordField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
    for (auto ib: bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coordField, node);
        *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateNP1), node) = std::cos(coordptr[0]);
        *stk::mesh::field_data(qField.field_of_state(stk::mesh::StateN), node) = std::cos(coordptr[0]);

        *stk::mesh::field_data(rhoField.field_of_state(stk::mesh::StateNP1), node) = 1.0;
        *stk::mesh::field_data(rhoField.field_of_state(stk::mesh::StateN), node) = 1.0;


        *stk::mesh::field_data(lidField, node) = node.local_offset();
        *stk::mesh::field_data(gidField, node) = bulk.identifier(node);
      }
    }

  }

  stk::ParallelMachine comm;
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  GlobalIdFieldType& gidField;
  stk::mesh::Field<int32_t>& lidField;
  stk::mesh::Field<double>& qField;
  stk::mesh::Field<double>& rhoField;
  stk::mesh::Field<double>& diffusivityField;
  stk::mesh::Field<double>& specificHeatField;

  std::string meshSpec{"generated:2x2x2"};
};

TEST_F(TpetraMeshManagerFixture, MeshIdDataTest)
{
  if (stk::parallel_machine_size(comm) > 8) return;
  auto data = determine_parallel_mesh_id_categorizations(bulk, gidField, meta.universal_part());
  if (stk::parallel_machine_size(comm) == 1) {
    ASSERT_EQ(data.maxOwnedRowId_, 27);
    ASSERT_EQ(data.ownedGids_.size(), 27u);
    ASSERT_EQ(data.maxSharedNotOwnedRowId_, 27u);
    ASSERT_EQ(data.sharedNotOwnedGids_.size(), 0u);
    ASSERT_EQ(data.localIds_.size(), 27u);
    ASSERT_FALSE(data.localIds_.empty());
  }

  if (stk::parallel_machine_size(comm) > 1) {
    ASSERT_LT(data.maxOwnedRowId_, 27);
    ASSERT_LT(data.ownedGids_.size(), 27u);
    ASSERT_GT(data.maxSharedNotOwnedRowId_, 0);
    ASSERT_GT(data.sharedNotOwnedGids_.size(), 0u);
    ASSERT_FALSE(data.localIds_.empty());
  }
}

TEST_F(TpetraMeshManagerFixture, check_entity_list)
{
  if (stk::parallel_machine_size(comm) > 8) return;

  auto data = determine_parallel_mesh_id_categorizations(bulk, gidField, meta.universal_part());
  ASSERT_FALSE(data.localIds_.empty());

  auto solCat = SolutionPointCategorizer(bulk, gidField, {}, {});
  auto entity_offsets = entity_offset_to_row_lid_map(bulk, gidField, solCat, meta.universal_part(), data.localIds_);

  for (int k = 1; k < 27; ++k) {
    EXPECT_EQ(entity_offsets(k-1), k - 1);
  }
}

}}
