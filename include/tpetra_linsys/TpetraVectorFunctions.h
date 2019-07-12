/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TpetraVectorFunctions_h
#define TpetraVectorFunctions_h

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include "element_promotion/NodeMapMaker.h"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"

#include "stk_mesh/base/FieldParallel.hpp"

namespace sierra { namespace nalu {

template <int p, typename TpetraVectorViewType>
nodal_scalar_array<DoubleType, p> gather_delta(
  int index,
  const elem_ordinal_view_t<p>& entToLID,
  const TpetraVectorViewType& xin)
{
  static constexpr int n1D = p + 1;
  nodal_scalar_array<DoubleType, p> guess;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          stk::simd::get_data(guess(k, j, i), n) = xin(entToLID(index, n, k, j, i), 0);
        }
      }
    }
  }
  return guess;
}

template <int p, typename TpetraVectorViewType>
nodal_scalar_array<DoubleType, p> gather_delta(
  int index,
  const elem_ordinal_view_t<p>& entToLID,
  const TpetraVectorViewType& xin,
  int d)
{
  static constexpr int n1D = p + 1;
  auto guess = la::zero<nodal_scalar_array<DoubleType, p>>();
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          stk::simd::get_data(guess(k, j, i), n) = xin(entToLID(index, n, k, j, i), d);
        }
      }
    }
  }
  return guess;
}


template <int p, typename TpetraVectorViewType>
nodal_vector_array<DoubleType, p> gather_vector_delta(
  int index,
  const elem_ordinal_view_t<p>& entToLID,
  const TpetraVectorViewType& xin)
{
  static constexpr int n1D = p + 1;
  auto guess = la::zero<nodal_vector_array<DoubleType, p>>();
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto lid = entToLID(index, n, k, j, i);
          stk::simd::get_data(guess(k, j, i, 0), n) = xin(lid, 0);
          stk::simd::get_data(guess(k, j, i, 1), n) = xin(lid, 1);
          stk::simd::get_data(guess(k, j, i, 2), n) = xin(lid, 2);
        }
      }
    }
  }
  return guess;
}



template <int p, typename TpetraVectorViewType>
void add_element_rhs_to_local_tpetra_vector(
  int index, const elem_ordinal_view_t<p>& entToLID,
  const nodal_scalar_array<DoubleType, p>& simdrhs,
  TpetraVectorViewType& yout)
{
  static constexpr int n1D = p + 1;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          yout(entToLID(index, n, k, j, i), 0) -= stk::simd::get_data(simdrhs(k, j, i), n);
        }
      }
    }
  }
}


template <int p, typename TpetraVectorViewType>
void add_element_rhs_to_local_tpetra_vector(
  int index,
  global_ordinal_type maxOwnedLID,
  global_ordinal_type maxSharedNotOwnedLid,
  const elem_ordinal_view_t<p>& entToLID,
  const nodal_scalar_array<DoubleType, p>& simdrhs,
  TpetraVectorViewType& yOwned,
  TpetraVectorViewType& yShared)
{
  static constexpr int n1D = p + 1;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto rowLid = entToLID(index, n, k, j, i);
          if (rowLid < maxOwnedLID) {
            yOwned(rowLid, 0) -= stk::simd::get_data(simdrhs(k, j, i), n);
          }
          else if (rowLid < maxSharedNotOwnedLid) {
            yShared(rowLid - maxOwnedLID, 0) -= stk::simd::get_data(simdrhs(k, j, i), n);
          }
        }
      }
    }
  }
}

template <int p, typename TpetraVectorViewType>
void add_element_rhs_to_local_tpetra_vector(
  int index,
  global_ordinal_type maxOwnedLID,
  global_ordinal_type maxSharedNotOwnedLid,
  const elem_ordinal_view_t<p>& entToLID,
  const nodal_scalar_array<DoubleType, p>& simdrhs,
  TpetraVectorViewType& yOwned,
  TpetraVectorViewType& yShared,
  int d)
{
  static constexpr int n1D = p + 1;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto rowLid = entToLID(index, n, k, j, i);
          if (rowLid < maxOwnedLID) {
            yOwned(rowLid, d) -= stk::simd::get_data(simdrhs(k, j, i), n);
          }
          else if (rowLid < maxSharedNotOwnedLid) {
            yShared(rowLid - maxOwnedLID, d) -= stk::simd::get_data(simdrhs(k, j, i), n);
          }
        }
      }
    }
  }
}

template <int p, typename TpetraVectorViewType>
void add_element_rhs_to_local_tpetra_vector(
  int index,
  global_ordinal_type maxOwnedLID,
  global_ordinal_type maxSharedNotOwnedLid,
  const elem_ordinal_view_t<p>& entToLID,
  const nodal_vector_array<DoubleType, p>& simdrhs,
  TpetraVectorViewType& yOwned,
  TpetraVectorViewType& yShared)
{
  static constexpr int n1D = p + 1;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0,0)); ++n) {
    for (int k = 0; k < n1D; ++k) {
      for (int j = 0; j < n1D; ++j) {
        for (int i = 0; i < n1D; ++i) {
          const auto rowLid = entToLID(index, n, k, j, i);
          if (rowLid < maxOwnedLID) {
            yOwned(rowLid, 0) -= stk::simd::get_data(simdrhs(k, j, i, 0), n);
            yOwned(rowLid, 1) -= stk::simd::get_data(simdrhs(k, j, i, 1), n);
            yOwned(rowLid, 2) -= stk::simd::get_data(simdrhs(k, j, i, 2), n);
          }
          else if (rowLid < maxSharedNotOwnedLid) {
            const auto offsetRow = rowLid - maxOwnedLID;
            yShared(offsetRow, 0) -= stk::simd::get_data(simdrhs(k, j, i, 0), n);
            yShared(offsetRow, 1) -= stk::simd::get_data(simdrhs(k, j, i, 1), n);
            yShared(offsetRow, 2) -= stk::simd::get_data(simdrhs(k, j, i, 2), n);
          }
        }
      }
    }
  }
}

template <int p, typename TpetraVectorViewType>
void add_face_rhs_to_local_tpetra_vector(
  int index,
  global_ordinal_type maxOwnedLID,
  global_ordinal_type maxSharedNotOwnedLid,
  const face_ordinal_view_t<p>& entToLID,
  const face_nodal_vector_array<DoubleType, p>& simdrhs,
  TpetraVectorViewType& yOwned,
  TpetraVectorViewType& yShared)
{
  static constexpr int n1D = p + 1;
  for (int n = 0; n < simdLen && valid_index(entToLID(index,n,0,0)); ++n) {
    for (int j = 0; j < n1D; ++j) {
      for (int i = 0; i < n1D; ++i) {
        const auto rowLid = entToLID(index, n, j, i);
        if (rowLid < maxOwnedLID) {
          yOwned(rowLid, 0) -= stk::simd::get_data(simdrhs(j, i, 0), n);
          yOwned(rowLid, 1) -= stk::simd::get_data(simdrhs(j, i, 1), n);
          yOwned(rowLid, 2) -= stk::simd::get_data(simdrhs(j, i, 2), n);
        }
        else if (rowLid < maxSharedNotOwnedLid) {
          const auto offsetRow = rowLid - maxOwnedLID;
          yShared(offsetRow, 0) -= stk::simd::get_data(simdrhs(j, i, 0), n);
          yShared(offsetRow, 1) -= stk::simd::get_data(simdrhs(j, i, 1), n);
          yShared(offsetRow, 2) -= stk::simd::get_data(simdrhs(j, i, 2), n);
        }
      }
    }
  }
}


template <typename TpetraViewType>
void update_solution(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& activeSelector,
  Kokkos::View<int*> nodeEntToLID,
  const TpetraViewType& xv,
  ScalarFieldType& qField)
{
  auto active_locally_owned = activeSelector & bulk.mesh_meta_data().locally_owned_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, active_locally_owned);
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      *stk::mesh::field_data(qField, node) -= xv(nodeEntToLID(node.local_offset()), 0);
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {&qField});
}

template <typename TpetraViewType>
void write_solution_to_field(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& activeSelector,
  Kokkos::View<int*> nodeEntToLID,
  const TpetraViewType& xv,
  ScalarFieldType& tmpField)
{
  auto active_locally_owned = activeSelector & bulk.mesh_meta_data().locally_owned_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, active_locally_owned);
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      *stk::mesh::field_data(tmpField, node) = -xv(nodeEntToLID(node.local_offset()), 0);
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {&tmpField});
}

template <typename TpetraViewType>
void write_solution_to_field(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& activeSelector,
  Kokkos::View<int*> nodeEntToLID,
  const TpetraViewType& xv,
  VectorFieldType& tmpField)
{
  auto active_locally_owned = activeSelector & bulk.mesh_meta_data().locally_owned_part();
  const auto& buckets = bulk.get_buckets(stk::topology::NODE_RANK, active_locally_owned);
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      auto* data = stk::mesh::field_data(tmpField, node);
      data[0] = -xv(nodeEntToLID(node.local_offset()), 0);
      data[1] = -xv(nodeEntToLID(node.local_offset()), 1);
      data[2] = -xv(nodeEntToLID(node.local_offset()), 2);
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {&tmpField});
}



} // namespace nalu
} // namespace Sierra

#endif

