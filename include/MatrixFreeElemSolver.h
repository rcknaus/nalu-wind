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
#include "ScratchViewslHO.h"

#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace sierra { namespace nalu { using TpetraOperator = Tpetra::Operator<double, int32_t, int64_t>; } }
namespace sierra { namespace nalu { using TpetraVector = Tpetra::Vector<double, int32_t, int64_t,
    Tpetra::Map<int32_t, int64_t>::node_type>; } }

namespace stk { namespace mesh { class Part; } }
namespace stk { namespace mesh { class Topology; } }

namespace sierra{
namespace nalu{

class MasterElement;

inline std::array<stk::mesh::Entity, simdLen>
load_simd_elems(const stk::mesh::Bucket& b, int bktIndex, int /* bucketLen */, int numSimdElems)
{
  std::array<stk::mesh::Entity, simdLen> simdElems;
  for (int simdElemIndex = 0; simdElemIndex < numSimdElems; ++simdElemIndex) {
    simdElems[simdElemIndex] = b[bktIndex*simdLen + simdElemIndex];
  }
  return simdElems;
}


inline int num_scalars_pre_req_data_HO(int nodesPerEntity, const ElemDataRequests& elemDataNeeded)
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

inline int calculate_shared_mem_bytes_per_thread_HO(int lhsSize, int rhsSize, int scratchIdsSize, int order, int dim, const ElemDataRequests& elemDataNeeded)
{
  int bytes_per_thread = ((rhsSize + lhsSize) * sizeof(double) + (2 * scratchIdsSize) * sizeof(int));
  bytes_per_thread += sizeof(double) * num_scalars_pre_req_data_HO(std::pow(order+1,dim), elemDataNeeded);
  bytes_per_thread *= 2*simdLen;
  return bytes_per_thread;
}


template <int p>
class ScratchViewsHOMF
{
  static constexpr int maxViews = 25;
public:
  using value_type = DoubleType;

  ScratchViewsHOMF(const ElemDataRequests& dataNeeded);

  value_type* get_scratch_view_ptr(const ScalarFieldType& field)
  {
    return fieldViewsScalar[field.mesh_meta_data_ordinal()].data();
  }

  value_type* get_scratch_view_ptr(const VectorFieldType& field)
  {
    return fieldViewsVector[field.mesh_meta_data_ordinal()].data();
  }

  value_type* get_scratch_view_ptr(const GenericFieldType& field) // this is not safe
  {
    return fieldViewsTensor[field.mesh_meta_data_ordinal()].data();
  }

  template <typename CompiledTimeSizedViewType, typename FieldType>
  CompiledTimeSizedViewType get_scratch_view(const FieldType& field)
  {
    return CompiledTimeSizedViewType(get_scratch_view_ptr(field));
  }

  int total_bytes() const { return num_bytes_required; }

  Kokkos::Array<nodal_scalar_view<p, value_type>, maxViews> get_fields_scalar();


  std::array<const stk::mesh::Entity*, simdLen> elemNodes{{}};
  int numSimdElems{simdLen};

private:

  Kokkos::Array<nodal_scalar_view<p, value_type>, maxViews> fieldViewsScalar{{}};
  Kokkos::Array<nodal_vector_view<p, value_type>, maxViews> fieldViewsVector{{}};
  Kokkos::Array<nodal_tensor_view<p, value_type>, maxViews> fieldViewsTensor{{}};

  int num_bytes_required{0};
};

template <int p> struct SharedMemDataHOMF
{
  constexpr int rhsSize = (p+1)*(p+1)*(p+1);
  SharedMemDataHOMF(
    const sierra::nalu::TeamHandleType& team,
    const stk::mesh::BulkData& bulk,
    const ElemDataRequests& dataNeeded)
  : simdPrereqData(team, bulk, dataNeeded)
  {
    simdrhs = get_shmem_view_1D<DoubleType>(team, rhsSize);
    rhs = get_shmem_view_1D<double>(team, rhsSize);
  }

  ScratchViewsHOMF<p> simdPrereqData;

  nodal_scalar_view<p, DoubleType> simdrhs;
  SharedMemView<double*> rhs;

  SharedMemView<int*> scratchIds;
  SharedMemView<int*> sortPermutation;
};

template<typename T>
ScratchViewsHOMF<T>::ScratchViewsHOMF(const TeamHandleType& team,
             const stk::mesh::BulkData& bulkData,
             int order, int /* dim */,
             const ElemDataRequests& dataNeeded)
{
  int numScalars = 0;
  const stk::mesh::MetaData& meta = bulkData.mesh_meta_data();
  unsigned numFields = meta.get_fields().size();
  fieldViews.resize(numFields);

  const FieldSet& neededFields = dataNeeded.get_fields_scalar();
  for(const FieldInfo& fieldInfo : neededFields) {
    ThrowAssert(fieldInfo.field->entity_rank() == stk::topology::NODE_RANK);
    unsigned scalarsDim1 = fieldInfo.scalarsDim1;

    if

    const int n1D = order + 1;
    if (scalarsDim1 == 1u) {

      fieldViews[fieldInfo.field->mesh_meta_data_ordinal()] = get_shmem_view_1D<T>(team, n1D * n1D * n1D);
      numScalars += n1D * n1D * n1D;
    }

    if (scalarsDim1 == 3u) {
      fieldViews[fieldInfo.field->mesh_meta_data_ordinal()] = get_shmem_view_1D<T>(team, 3 * n1D * n1D * n1D);
      numScalars += 3 * n1D * n1D * n1D;
    }
  }
  num_bytes_required += numScalars * sizeof(T);
}

template <int p, typename KernelType, bool VectorSystem = false>
class MatrixFreeElemSolver : Tpetra::Operator<double, int32_t, int64_t>
{
  DeclareCVFEMTypeDefs(CVFEMViews<p>);
  static constexpr int dim = 3;
  static constexpr int n1D = p + 1;
  static constexpr int rhsSize = n1D*n1D*n1D;
public:
  MatrixFreeElemSolver( stk::mesh::Part* part,
    unsigned nodesPerEntity);
  virtual ~MatrixFreeElemSolver() = default;
  virtual void initialize_connectivity();
  virtual void execute();


  virtual void apply (
    const TpetraVector& X,
    TpetraVector& Y,
    Teuchos::ETransp mode = Teuchos::NO_TRANS,
    double alpha = 1.0,
    double beta = 0.0
  ) const;

  virtual Teuchos::RCP<const Tpetra::Map<int32_t, int64_t>> getDomainMap() const
  {
    return rowMap_;
  }

  virtual Teuchos::RCP<const Tpetra::Map<int32_t, int64_t>> getRangeMap() const
  {
    return rowMap_;
  }




  void initialize_for_iteration()
  {
    // the pre-amble work in the loop
  }

  template<typename LambdaFunction>
  void run_algorithm(stk::mesh::BulkData& bulk, const TpetraVector& Xin, const TpetraVector& Yout)
  {




    const stk::mesh::MetaData& meta_data = bulk.mesh_meta_data();
    const int lhsSize = rhsSize_*rhsSize_;
    const int scratchIdsSize = rhsSize_;

   const int bytes_per_team = 0;
   const int bytes_per_thread = calculate_shared_mem_bytes_per_thread_HO(lhsSize, rhsSize_, scratchIdsSize,
     polyOrder_, meta_data.spatial_dimension(), dataNeededByKernels_);
   stk::mesh::Selector elemSelector =
           meta_data.locally_owned_part()
         & stk::mesh::selectUnion(partVec_)
         & !realm_.get_inactive_selector();
 
   stk::mesh::BucketVector const& elem_buckets = bulk.get_buckets(entityRank_, elemSelector);
   auto team_exec = sierra::nalu::get_host_team_policy(elem_buckets.size(), bytes_per_team, bytes_per_thread);

   kernel.setup()

   Kokkos::parallel_for(team_exec, [&](const sierra::nalu::TeamHandleType& team)
   {
     const stk::mesh::Bucket& b = *elem_buckets[team.league_rank()];
 
     ScratchViewsHOMF<p> simdPrereqData;
     nodal_scalar_view simdrhs;
     SharedMemView<double*> rhs;

     const size_t bucketLen = b.size();
     const size_t simdBucketLen = get_num_simd_groups(bucketLen);
 
     Kokkos::parallel_for(Kokkos::TeamThreadRange(team, simdBucketLen), [&](const size_t& bktIndex)
     {
       const int numSimdElems = get_length_of_next_simd_group(bktIndex, bucketLen);
       const auto simdElems = load_simd_elems(b, bktIndex, bucketLen, numSimdElems);

       fill_pre_req_data(dataNeededByKernels_, bulk, simdElems, numSimdElems, simdPrereqData);
       set_zero(simdrhs.data(), rhsSize);

       kernel.execute();

       for(int j=0; j < simdPrereqData.numSimdElems; ++j) {
         for (int i = 0; i < rhsSize; ++i) {
           const int rowLid = entityToLID_[simdPrereqData.elemNodes[perm_[i]].local_offset()]; // feels like a lot of indirection
           if(rowLid < maxOwnedRowId_) {
             if (forceAtomic) {
               Kokkos::atomic_add(&Yout(rowLid,0), cur_rhs);
             }
             else {
               Yout(rowLid,0) += cur_rhs;
             }
           }
           else if (rowLid < maxSharedNotOwnedRowId_) {
             int32_t actualLocalId = rowLid - maxOwnedRowId_;

             if (forceAtomic) {
               Kokkos::atomic_add(&sharedNotOwnedLocalRhs_(actualLocalId,0), cur_rhs);
             }
             else {
               sharedNotOwnedLocalRhs_(actualLocalId,0) += cur_rhs;
             }
           }
         }
       }
     });
   });
  }

private:
  void extract_permuted_vector_lanes(
    const SharedMemView<DoubleType*>& simdrhs,
    const Kokkos::View<int*>& perm,
    int simdIndex,
    SharedMemView<double**>& lhs,
    SharedMemView<double*>& rhs)
  {
    for (int j = 0; j < rhsSize; ++j) {
      rhs[perm[j]] = stk::simd::get_data(simdrhs[j], simdIndex);
    }
  }

  void gather_elem_node_field(
    Kokkos::View<int*>& perm,
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
    Kokkos::View<int*>& perm,
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
      auto& field = *static_cast<ScalarFieldType*>(fieldInfo.field);
      gather_elem_node_field(
        perm,
        field,
        prereqData.elemNodes,
        prereqData.numSimdElems,
        prereqData.get_scratch_view(field)
      );
    }
  }

private:
  KernelType kernel;
  stk::mesh::PartVector parts;
  Kokkos::View<int32_t*> entityToLID_;
  Teuchos::RCP<Tpetra::Map<int32_t, int64_t>> rowMap_;
  int32_t maxOwnedRowId_;
  int32_t maxSharedNotOwnedRowId_;

  const Kokkos::View<int*> perm_;
  ElemDataRequests dataNeededByKernels_;
};

} // namespace nalu
} // namespace Sierra

#endif

