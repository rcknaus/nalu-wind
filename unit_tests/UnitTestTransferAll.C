#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <stk_io/StkMeshIoBroker.hpp>


#include <master_element/MasterElement.h>
#include <master_element/Hex8GeometryFunctions.h>
#include <master_element/MasterElementUtils.h>

#include <xfer/ElemToNodeTransfer.h>
#include <nalu_make_unique.h>

#include <random>
#include <array>

#include "UnitTestUtils.h"

namespace  sierra { namespace nalu { namespace {

  struct LinearField
  {
    LinearField(double in_a, const double* in_b) :a(in_a)
    {
      b[0] = in_b[0];
      b[1] = in_b[1];
      b[2] = in_b[2];
    }

    LinearField(const LinearField& src) :  a(src.a)
    {
      b[0] = src.b[0];
      b[1] = src.b[1];
      b[2] = src.b[2];
    }

    double operator()(const double* x) { return (a + b[0] * x[0] + b[1] * x[1] + b[2] * x[2]); }

    const double a;
    double b[3];
  };

  LinearField make_random_linear_field(std::mt19937& rng)
  {
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    std::array<double, 3> coeffs = {{ coeff(rng), coeff(rng), coeff(rng) }};
    return LinearField(coeff(rng), coeffs.data());
  }


  std::pair<std::unique_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>> generate_hex8_mesh(
    int n,
    LinearField x,
    std::string fieldName = "scalar_field"
   )
  {
    stk::ParallelMachine comm = MPI_COMM_WORLD;
    constexpr int dim = 3;

    auto metaPtr = make_unique<stk::mesh::MetaData>(dim);
    auto bulkPtr = make_unique<stk::mesh::BulkData>(*metaPtr, comm);

    auto& scalarField = metaPtr->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, fieldName);

    double one = 1.0;
    stk::mesh::put_field_on_mesh(scalarField, metaPtr->universal_part(), 1, &one);


    std::string n_string = std::to_string(n);
    std::string spec = "generated:" + n_string + "x" + n_string + "x" + n_string;
    unit_test_utils::fill_hex8_mesh(spec, *bulkPtr);

    auto& coordField = *static_cast<const VectorFieldType*>(metaPtr->coordinate_field());
    for (const auto* ib : bulkPtr->get_buckets(stk::topology::NODE_RANK, metaPtr->universal_part())) {
     for (auto node : *ib) {
        for (int d = 0; d < dim; ++d) {
          double coord = stk::mesh::field_data(coordField, node)[d];
          stk::mesh::field_data(coordField, node)[d] = coord/n;
        }
      }
    }

    for (const auto* ib : bulkPtr->get_buckets(stk::topology::NODE_RANK, metaPtr->universal_part())) {
     for (auto node : *ib) {
       *stk::mesh::field_data(scalarField, node) = x(stk::mesh::field_data(coordField, node))+0;
      }
    }
    return std::make_pair(std::move(metaPtr), std::move(bulkPtr));
  }

  void test_interpolative_transfer(int n1, int n2)
  {
    std::mt19937 rng;
    rng.seed(std::random_device()());
    LinearField x = make_random_linear_field(rng);
    LinearField y = make_random_linear_field(rng);

    auto mesh1 = generate_hex8_mesh(n1, x);
    auto mesh2 = generate_hex8_mesh(n2, y);


    auto& scalarField = *static_cast<const ScalarFieldType*>(mesh2.first->get_field(stk::topology::NODE_RANK, "scalar_field"));

    transfer::transfer_all(*mesh1.second, mesh1.first->universal_part(), *mesh2.second, mesh2.first->universal_part());

    auto& coordField = *static_cast<const VectorFieldType*>(mesh2.first->coordinate_field());

    for (const auto* ib : mesh2.second->get_buckets(stk::topology::NODE_RANK, mesh2.first->universal_part())) {
      for (auto node : *ib) {
        double meshVal = *stk::mesh::field_data(scalarField, node);
        double linearValFromMesh1 = x(stk::mesh::field_data(coordField, node));
        EXPECT_NEAR(meshVal, linearValFromMesh1, 1.0e-10);
      }
    }
  }

}

TEST(transfer_all, same_mesh)
{
   test_interpolative_transfer(9,9);
}

TEST(transfer_all, coarse_to_fine)
{
   test_interpolative_transfer(2,4);
}

TEST(transfer_all, fine_to_coarse)
{
   test_interpolative_transfer(4,2);
}




std::pair<std::unique_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>>
read_mesh(std::string meshName, std::set<std::string> fieldNames, stk::ParallelMachine comm, int dim)
{
  auto metaPtr = make_unique<stk::mesh::MetaData>(dim);
  auto bulkPtr = make_unique<stk::mesh::BulkData>(*metaPtr, comm);

  stk::io::StkMeshIoBroker io(comm);
  io.set_bulk_data(*bulkPtr);
  io.add_mesh_database(meshName, stk::io::READ_RESTART);
  io.create_input_mesh();

  for (auto fieldName : fieldNames) {
    auto& scalarField = metaPtr->declare_field<ScalarFieldType>(stk::topology::NODE_RANK, fieldName);
    stk::mesh::put_field_on_mesh(scalarField, metaPtr->universal_part(), 1u, nullptr);
    io.add_input_field({*stk::mesh::get_field_by_name(fieldName,*metaPtr), fieldName});
  }

  io.populate_bulk_data();

  std::vector<stk::io::MeshField> missingFields;
  io.read_defined_input_fields(1000.0, &missingFields);
  io.populate_field_data();

  return std::make_pair(std::move(metaPtr), std::move(bulkPtr));
}

TEST(transfer_all, read_restart_fields)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;

  std::string fileName = "test.rst";
  std::mt19937 rng;
  rng.seed(std::random_device()());

  auto fieldName = "scalar_field";
  auto x = make_random_linear_field(rng);
  {
    auto mesh = generate_hex8_mesh(1, x, fieldName);
    auto& meta = *mesh.first;
    auto& bulk = *mesh.second;

    stk::io::StkMeshIoBroker io(comm);
    io.set_bulk_data(bulk);
    auto idx = io.create_output_mesh(fileName, stk::io::WRITE_RESTART);

    io.add_field(idx, *stk::mesh::get_field_by_name(fieldName, meta), fieldName);
    io.add_global(idx, "timeStepNm1",  0, stk::util::ParameterType::DOUBLE);
    io.add_global(idx, "timeStepCount", 0, stk::util::ParameterType::INTEGER);

    double deltat = 0.1;
    int tstep = 100;
    io.begin_output_step(idx, 10.0);
    io.write_defined_output_fields(idx);
    io.write_global(idx, "timeStepNm1", deltat);
    io.write_global(idx, "timeStepCount", tstep);
    io.end_output_step(idx);
  }
  auto mesh = read_mesh(fileName, {fieldName}, comm, 3);
  auto& meta = *mesh.first;
  auto& bulk = *mesh.second;

  auto& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());
  auto& scalarField = *meta.get_field<ScalarFieldType>(stk::topology::NODE_RANK, fieldName);

  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK,meta.universal_part())) {
    for (auto node : *ib) {
      double meshVal = *stk::mesh::field_data(scalarField, node);
      double linearValFromMesh1 = x(stk::mesh::field_data(coordField, node));
      EXPECT_NEAR(meshVal, linearValFromMesh1, 1.0e-10);
    }
  }
}



}
}



