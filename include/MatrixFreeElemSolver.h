/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MatrixFreeElemSolver_h
#define MatrixFreeElemSolver_h

#include <Realm.h>
#include <SolverAlgorithm.h>
#include <ElemDataRequests.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <CopyAndInterleave.h>
#include <FieldTypeDef.h>

#include "CVFEMTypeDefs.h"
#include "ScratchViewsHOMF.h"

#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace sierra { namespace nalu { using TpetraOperator = Tpetra::Operator<double, int32_t, int64_t>; } }
namespace sierra { namespace nalu { using TpetraVector = Tpetra::MultiVector<double, int32_t, int64_t,
    Tpetra::Map<int32_t, int64_t>::node_type>; } }

namespace stk { namespace mesh { class Part; } }
namespace stk { namespace mesh { class Topology; } }

namespace sierra{
namespace nalu{

template <int p, typename KernelType>
class MatrixFreeElemSolver : Tpetra::Operator<double, int32_t, int64_t>
{
  DeclareCVFEMTypeDefs(CVFEMViews<p>);
  static constexpr int dim = 3;
  static constexpr int n1D = p + 1;
  static constexpr int rhsSize = n1D * n1D * n1D * KernelType::ndof;
  static constexpr int npe = n1D * n1D * n1D;
  static constexpr bool forceAtomic = false;
public:
  MatrixFreeElemSolver(stk::mesh::BulkData& bulk, TimeIntegrator& ti, ElemDataRequests& dataNeeded, KernelType kernel)
: kernel_(kernel), bulk_(bulk), ti_(ti), dataNeeded_(dataNeeded) {}
  virtual ~MatrixFreeElemSolver() = default;
  virtual void initialize_connectivity();
  virtual void execute();

  virtual void apply (
    const TpetraVector& X,
    TpetraVector& Y,
    Teuchos::ETransp mode = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0
  ) const
  {
//    run_algorithm(bulk_, ti_, X, Y);
  }


  virtual Teuchos::RCP<const Tpetra::Map<int32_t, int64_t>> getDomainMap() const
  {
    return rowMap_;
  }

  virtual Teuchos::RCP<const Tpetra::Map<int32_t, int64_t>> getRangeMap() const
  {
    return rowMap_;
  }

  int calculate_required_shared_memory(const ElemDataRequests& dataNeeded)
  {
    const int scalars_required = rhsSize + num_scalars_pre_req_data_HO(rhsSize, dataNeeded);
    const int byte_multiplier = 2 * sizeof(double) * simdLen;
    return scalars_required * byte_multiplier;
  }

  void initialize_for_iteration()
  {
    // the pre-amble work in the loop
  }

  void run_algorithm(stk::mesh::BulkData& bulk, const TimeIntegrator& ti, const TpetraVector& Xin, TpetraVector& Yout)
  {
    const stk::mesh::MetaData& meta_data = bulk.mesh_meta_data();

    const int bytes_per_team = 0;
    const int bytes_per_thread = calculate_required_shared_memory(dataNeeded_);
    const auto& elem_buckets = bulk.get_buckets(stk::topology::ELEM_RANK, selector_);
    auto team_exec = sierra::nalu::get_host_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

    kernel_.setup(ti);
    auto x1D = Xin.get1dView();
    auto y1D = Yout.get1dViewNonConst();


    Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
    {
      const stk::mesh::Bucket& b = *elem_buckets[team.league_rank()];

      ScratchViewsHOMF<p> simdData(team, bulk, dataNeeded_);
      nodal_scalar_view simdrhs;

      const size_t bucketLen = b.size();
      const size_t simdBucketLen = get_num_simd_groups(bucketLen);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
      {
        const int numSimdElems = get_length_of_next_simd_group(bktIndex, bucketLen);
        const auto simdElems = load_simd_elems(b, bktIndex, bucketLen, numSimdElems);

        fill_pre_req_data(dataNeeded_, bulk, perm_, simdElems, numSimdElems, simdData);

        DoubleType* simdrhs_ptr = simdrhs.data();
        for (int j = 0; j < rhsSize; ++j) {
          simdrhs_ptr[j] = 0;
        }

        auto sol_field_data = simdData.get_scratch_view_ptr(kernel_.solution_field());
        for (int j = 0; j < simdData.numSimdElems; ++j) {
          for (int i = 0; i < rhsSize; ++i) {
            const int rowLid = entityToLID_[simdData.elemNodes[j][perm_[i]].local_offset()];
            sol_field_data[i][j] += x1D[rowLid];
          }
        }

        kernel_.executemf(simdrhs, simdData);

        for (int j = 0; j < simdData.numSimdElems; ++j) {
          for (int i = 0; i < rhsSize; ++i) {
            const int rowLid = entityToLID_[simdData.elemNodes[j][perm_[i]].local_offset()];
            y1D[rowLid] += simdrhs_ptr[i][j];
          }
        }
      });
    });
  }

private:
  void update_solution_field()
  {

  }

  void gather_elem_node_field(
    const Kokkos::View<int*>& perm,
    const ScalarFieldType& field,
    const std::array<const stk::mesh::Entity*, simdLen>& elemNodes,
    int numSimdElems,
    nodal_scalar_view& shmemView) const
  {
    constexpr int n1D = p + 1;
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const int permutationIndex = perm((k * n1D + j) * n1D + i);
          for (int simdIndex = 0; simdIndex < numSimdElems; ++simdIndex) {
            stk::simd::set_data(shmemView(k, j, i), simdIndex,
              *static_cast<const double*>(stk::mesh::field_data(field, elemNodes[simdIndex][permutationIndex])));
          }
        }
      }
    }
  }

  void fill_pre_req_data(
    ElemDataRequests& dataNeeded,
    const stk::mesh::BulkData& bulkData,
    const Kokkos::View<int*>& perm,
    const std::array<stk::mesh::Entity, simdLen>& elems,
    int numSimdElems,
    ScratchViewsHOMF<p>& prereqData)
  {
    for (int simdIndex = 0; simdIndex < numSimdElems; ++simdIndex) {
      prereqData.elemNodes[simdIndex] = bulkData.begin_nodes(elems[simdIndex]);
    }
    prereqData.numSimdElems = numSimdElems;

    const FieldSet& neededFields = dataNeeded.get_fields();
    for (const FieldInfo& fieldInfo : neededFields) {
      const auto& field = *static_cast<const ScalarFieldType*>(fieldInfo.field);
      auto view = prereqData.get_scratch_view(field);
      gather_elem_node_field(
        perm,
        field,
        prereqData.elemNodes,
        prereqData.numSimdElems,
        view
      );
    }
  }


  std::array<stk::mesh::Entity, simdLen>
  load_simd_elems(const stk::mesh::Bucket& b, int bktIndex, int /* bucketLen */, int numSimdElems)
  {
    std::array<stk::mesh::Entity, simdLen> simdElems;
    for (int simdElemIndex = 0; simdElemIndex < numSimdElems; ++simdElemIndex) {
      simdElems[simdElemIndex] = b[bktIndex*simdLen + simdElemIndex];
    }
    return simdElems;
  }


  int num_scalars_pre_req_data_HO(int nodesPerEntity, const ElemDataRequests& elemDataNeeded)
  {
    int numScalars = 0;
    const FieldSet& neededFields = elemDataNeeded.get_fields();
    for(const FieldInfo& fieldInfo : neededFields) {
      stk::mesh::EntityRank fieldEntityRank = fieldInfo.field->entity_rank();
      unsigned scalarsPerEntity = fieldInfo.scalarsDim1;
      unsigned entitiesPerElem = fieldEntityRank==stk::topology::NODE_RANK ? nodesPerEntity : 1;
      ThrowRequire(entitiesPerElem > 0);
      if (fieldInfo.scalarsDim2 > 1) {
        scalarsPerEntity *= fieldInfo.scalarsDim2;
      }
      numScalars += entitiesPerElem*scalarsPerEntity;
    }
    return numScalars;
  }

  int calculate_shared_mem_bytes_per_thread_HO(int lhsSize, int rhsSize, int scratchIdsSize, int order, int dim, const ElemDataRequests& elemDataNeeded)
  {
    int bytes_per_thread = ((rhsSize + lhsSize) * sizeof(double) + (2 * scratchIdsSize) * sizeof(int));
    bytes_per_thread += sizeof(double) * num_scalars_pre_req_data_HO(std::pow(order+1,dim), elemDataNeeded);
    bytes_per_thread *= 2*simdLen;
    return bytes_per_thread;
  }

  KernelType kernel_;
  stk::mesh::BulkData& bulk_;
  TimeIntegrator& ti_;
  ElemDataRequests dataNeeded_;

  Kokkos::View<int32_t*> entityToLID_;
  Teuchos::RCP<Tpetra::Map<int32_t, int64_t>> rowMap_;
  stk::mesh::Selector selector_;

  const Kokkos::View<int*> perm_;

};

} // namespace nalu
} // namespace Sierra

#endif

