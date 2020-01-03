#include "matrix_free/ConductionGatheredFieldManager.h"

#include "matrix_free/ConductionFields.h"
#include "StkConductionFixture.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_ngp/NgpFieldParallel.hpp"

#include <Kokkos_Macros.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <memory>
#include <stk_math/StkMath.hpp>
#include <stk_ngp/NgpForEachEntity.hpp>
#include <stk_simd/Simd.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

class ConductionGatheredFieldManagerFixture : public ::ConductionFixture
{
protected:
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;
  ConductionGatheredFieldManagerFixture()
    : ConductionFixture(nx, scale),
      field_gather(meta, mesh, fm, meta.universal_part(), {})
  {
    auto& coordField =
      *meta.get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "coordinates");
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coordField, node);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) = coordptr[0];
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) = coordptr[0];
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) = coordptr[0];
        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = 1.0;
        *stk::mesh::field_data(lambda_field, node) = 1.0;
      }
    }
  }
  ConductionGatheredFieldManager<order> field_gather;
};

TEST_F(ConductionGatheredFieldManagerFixture, gather_all)
{
  field_gather.gather_all();
  auto residual_fields = field_gather.get_residual_fields();
  EXPECT_TRUE(residual_fields.volume_metric.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qm1.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qp0.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qp1.extent_int(0) > 1);
}

TEST_F(ConductionGatheredFieldManagerFixture, swap_states)
{
  field_gather.gather_all();
  auto residual_fields = field_gather.get_residual_fields();
  auto qm1_label = residual_fields.qm1.label();
  auto qp0_label = residual_fields.qp0.label();
  auto qp1_label = residual_fields.qp1.label();

  field_gather.swap_states();
  residual_fields = field_gather.get_residual_fields();
  EXPECT_EQ(residual_fields.qm1.label(), qp0_label);
  EXPECT_EQ(residual_fields.qp0.label(), qp1_label);
  EXPECT_EQ(residual_fields.qp1.label(), qm1_label);
}

namespace {
template <int p>
double
sum_field(scalar_view<p> qp1)
{
  double sum_prev = 0;
  Kokkos::parallel_reduce(
    qp1.extent_int(0),
    KOKKOS_LAMBDA(int index, double& sumval) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < simd_len; ++n) {
              sumval += stk::simd::get_data(qp1(index, k, j, i), n);
            }
          }
        }
      }
    },
    sum_prev);
  return sum_prev;
}

void
set_field(
  const ngp::Mesh& mesh,
  stk::mesh::Selector selector,
  ngp::Field<double>& field,
  double val)
{
  ngp::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(ngp::Mesh::MeshIndex mi) { field.get(mi, 0) = val; });
}

void
double_field(
  const ngp::Mesh& mesh,
  stk::mesh::Selector selector,
  ngp::Field<double>& field)
{
  ngp::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(ngp::Mesh::MeshIndex mi) { field.get(mi, 0) *= 2; });
}

} // namespace

TEST_F(ConductionGatheredFieldManagerFixture, update_solution)
{
  auto sol_field =
    fm.get_field<double>(meta
                           .get_field<stk::mesh::Field<double>>(
                             stk::topology::NODE_RANK, conduction_info::q_name)
                           ->field_state(stk::mesh::StateNP1)
                           ->mesh_meta_data_ordinal());
  set_field(mesh, meta.universal_part(), sol_field, 1);

  field_gather.gather_all();

  auto sum_prev = sum_field<order>(field_gather.get_residual_fields().qp1);

  double_field(mesh, meta.universal_part(), sol_field);
  field_gather.update_solution_fields();
  auto sum_post = sum_field<order>(field_gather.get_residual_fields().qp1);

  EXPECT_DOUBLE_EQ(sum_prev, sum_post / 2);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
