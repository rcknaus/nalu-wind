/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SparsifiedLaplacian_h
#define SparsifiedLaplacian_h

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/ConductionKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

namespace sierra { namespace nalu {

class TpetraLinearSystem;

template <int p> struct SparsifiedLaplacianInterior
{
  SparsifiedLaplacianInterior(
    const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    TpetraLinearSystem& linsys,
    const VectorFieldType& coordField);

  void initialize();
  void initialize_connectivity();
  void compute_lhs_simd();
  void compute_lhs();

private:
  const stk::mesh::BulkData& bulk_;
  stk::mesh::Selector selector_;
  TpetraLinearSystem& linsys_;
  const VectorFieldType& coordField_;

  ko::vector_view<p> coords_;
  elem_entity_view_t<p> entities_;
};

} // namespace nalu
} // namespace Sierra

#endif

