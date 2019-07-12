#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <SimdFieldGather.h>
#include <CVFEMVolumes.h>
#include <CVFEMMappedAreas.h>

#include <kernel/ScalarDiffHOElemKernel.h>
#include <FieldTypeDef.h>
#include <Teuchos_ArrayRCP.hpp>
#include <nalu_make_unique.h>

#include <UnitTestUtils.h>


#include <TimeIntegrator.h>
#include <SolutionOptions.h>
#include <tpetra_linsys/TpetraMeshManager.h>
#include <tpetra_linsys/SolutionPointCategorizer.h>

#include "element_promotion/PromotedPartHelper.h"

#include "UnitTestUtils.h"


namespace sierra { namespace nalu {

static constexpr int p = 2;

static constexpr double scalar_val_tmp = -1.23;
static constexpr double scalar_val = 2.43215;
static constexpr double vec_val = 3.0;

class SimdFieldGatherFixture : public ::testing::Test
{
protected:
  SimdFieldGatherFixture()
    : comm(MPI_COMM_WORLD),
      meta(3u), bulk(meta, comm, stk::mesh::BulkData::NO_AUTO_AURA),
      scalField(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "scalar_field")),
      scalFieldTmp(meta.declare_field<stk::mesh::Field<double>>(stk::topology::NODE_RANK, "scalar_field_tmp")),
      vecField(meta.declare_field<stk::mesh::Field<double, stk::mesh::Cartesian>>(stk::topology::NODE_RANK, "vector_field")),
      gidField(meta.declare_field<GlobalIdFieldType>(stk::topology::NODE_RANK, "global_id_field"))

  {
    std::vector<double> zeros = {1,1,1};
    stk::mesh::EntityId zeroId = 0u;

    stk::mesh::put_field_on_mesh(scalField, meta.universal_part(), 1, zeros.data());
    stk::mesh::put_field_on_mesh(scalFieldTmp, meta.universal_part(), 1, zeros.data());

    stk::mesh::put_field_on_mesh(vecField, meta.universal_part(), 3u, zeros.data());
    stk::mesh::put_field_on_mesh(gidField, meta.universal_part(), 1, &zeroId);

    unit_test_utils::fill_and_promote_hex_mesh(meshSpec, bulk, p);
    activeSelector_ = (p==1) ? meta.universal_part() : stk::mesh::selectUnion(only_super_elem_parts(meta.get_parts()));

    for (auto ib: bulk.get_buckets(stk::topology::NODE_RANK, activeSelector_)) {
      for (auto node : *ib) {
        stk::mesh::field_data(scalField, node)[0] = scalar_val;
        stk::mesh::field_data(scalFieldTmp, node)[0] = scalar_val_tmp;

        stk::mesh::field_data(vecField, node)[0] = vec_val;
        stk::mesh::field_data(vecField, node)[1] = vec_val;
        stk::mesh::field_data(vecField, node)[2] = vec_val;
        stk::mesh::field_data(gidField, node)[0] = bulk.identifier(node);
      }
    }

//    auto data = determine_parallel_mesh_id_categorizations(bulk, gidField, activeSelector_);
//    auto solCat = SolutionPointCategorizer(bulk, gidField, {}, {});
//    entity_offsets = element_entity_offset_to_gid_map<p>(bulk, activeSelector_,
//      entity_offset_to_row_lid_map(bulk, gidField, solCat, activeSelector_, data.localIds_)
//    );
    entity_offsets = element_entity_offset_to_gid_map<p>(bulk, activeSelector_, serial_entity_offset_to_row_lid_map(bulk, activeSelector_));
  }


  const VectorFieldType& coord_field() { return* static_cast<const VectorFieldType*>(meta.coordinate_field()); }

  stk::ParallelMachine comm;
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  stk::mesh::Field<double>& scalField;
  stk::mesh::Field<double>& scalFieldTmp;

  stk::mesh::Field<double, stk::mesh::Cartesian>& vecField;
  GlobalIdFieldType& gidField;
  elem_ordinal_view_t<p> entity_offsets;
  std::string meshSpec{"generated:2x2x2"};
  stk::mesh::Selector activeSelector_;
};

TEST_F(SimdFieldGatherFixture, gather_scalar)
{
  auto view = gather_field<p>(bulk, activeSelector_, scalField);
  for (int e = 0; e < view.extent_int(0); ++e) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          EXPECT_DOUBLETYPE_EQ(view(e, k,j,i), scalar_val);
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, gather_vector)
{
  auto view = gather_field<p>(bulk,activeSelector_, vecField);
  for (int e = 0; e < view.extent_int(0); ++e) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          for (int d = 0; d < 3; ++d) {
            EXPECT_DOUBLETYPE_EQ(view(e, k, j, i, d), vec_val);
          }
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, volumes_p1)
{
  const auto& coordField = *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
  auto vols = volumes<p>(gather_field<p>(bulk,activeSelector_,coordField));
  if (p == 1) {
    for (int e = 0; e < vols.extent_int(0); ++e) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int d = 0; d < 3; ++d) {
              EXPECT_DOUBLETYPE_EQ(vols(e, k, j, i, d), 0.125);
            }
          }
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, scaled_volumes_p1)
{
  const auto& coordField = *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
  auto vols = volumes<p>(gather_field<p>(bulk, activeSelector_, scalField), gather_field<p>(bulk, activeSelector_, coordField));
  if (p == 1) {

    for (int e = 0; e < vols.extent_int(0); ++e) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int d = 0; d < 3; ++d) {
              EXPECT_DOUBLETYPE_EQ(vols(e, k, j, i, d), DoubleType(scalar_val * 0.125));
            }
          }
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, scaled_mesh_volumes_p1)
{
  const auto& coordField = *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
  stk::mesh::field_scale(std::cbrt(scalar_val), coordField);
  auto vols = volumes<p>(gather_field<p>(bulk,activeSelector_, coordField));
  if (p == 1) {
    for (int e = 0; e < vols.extent_int(0); ++e) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int d = 0; d < 3; ++d) {
              EXPECT_DOUBLETYPE_EQ(vols(e, k, j, i, d), DoubleType(scalar_val * 0.125));
            }
          }
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, mapped_areas_v_p1)
{
  const auto& coordField = *static_cast<const VectorFieldType*>(bulk.mesh_meta_data().coordinate_field());
  auto mapped_area = mapped_areas<p>(gather_field<p>(bulk, activeSelector_, coordField));
  if (p == 1) {
    for (int e = 0; e < mapped_area.extent_int(0); ++e) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p; ++i) {
            for (int di = 0; di < 3; ++di) {
              EXPECT_DOUBLETYPE_EQ(mapped_area(e, 0, k, j, i, di), (di == 0) ? -0.5 : 0);
            }
          }
        }
      }

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int di = 0; di < 3; ++di) {
              EXPECT_DOUBLETYPE_EQ(mapped_area(e, 1, k, j, i, di), (di == 1) ? -0.5 : 0);
            }
          }
        }
      }

      for (int k = 0; k < p; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int di = 0; di < 3; ++di) {
              EXPECT_DOUBLETYPE_EQ(mapped_area(e, 2, k, j, i, di), (di == 2) ? -0.5 : 0);
            }
          }
        }
      }
    }
  }
}

TEST_F(SimdFieldGatherFixture, gather_scatter_are_inverses)
{
  auto elem_scalar_field = gather_field<p>(bulk,activeSelector_, scalField);
  write_to_stk_field<p>(bulk, entity_offsets, elem_scalar_field, scalFieldTmp);
  for (auto ib: bulk.get_buckets(stk::topology::NODE_RANK, activeSelector_)) {
    for (auto node : *ib) {
      ASSERT_DOUBLE_EQ(*stk::mesh::field_data(scalField, node), *stk::mesh::field_data(scalFieldTmp, node));
    }
  }
}


TEST_F(SimdFieldGatherFixture, check_doubled_field)
{

  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);


  for (const auto* ib: bulk.get_buckets(stk::topology::NODE_RANK, activeSelector_)) {
    for (auto node : *ib) {
      stk::mesh::field_data(scalField, node)[0] = coeff(rng);
    }
  }
  stk::mesh::field_copy(scalField, scalFieldTmp);

  auto elem_scal_field = gather_field<p>(bulk, activeSelector_, scalField);
  for (int e = 0; e < elem_scal_field.extent_int(0); ++e) {
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          elem_scal_field(e,k,j,i) *= 2;
        }
      }
    }
  }
  write_to_stk_field<p>(bulk, entity_offsets, elem_scal_field, scalField);

  for (const auto* ib: bulk.get_buckets(stk::topology::NODE_RANK, activeSelector_)) {
    for (auto node : *ib) {
      ASSERT_DOUBLE_EQ(stk::mesh::field_data(scalField, node)[0], 2*stk::mesh::field_data(scalFieldTmp, node)[0]);
    }
  }
}



}}
