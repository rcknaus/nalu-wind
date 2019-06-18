/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumDiagonal_h
#define MomentumDiagonal_h

#include "SimdFieldGather.h"
#include "CVFEMMappedAreas.h"
#include "CVFEMVolumes.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"

#include "MatrixFreeTypes.h"

#include "kernel/MomentumKernel.h"
#include "tpetra_linsys/TpetraVectorFunctions.h"

namespace sierra { namespace nalu {

class TpetraLinearSystem;

template <int p> struct MomentumInteriorDiagonal
{
  using op = momentum_element_residual<p>;

//  explicit MomentumInteriorDiagonal(TpetraLinearSystem& linsys);

  MomentumInteriorDiagonal(
    const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    TpetraLinearSystem& linsys,
    const VectorFieldType& coordField);

  void initialize(
    const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    const VectorFieldType& coordField);

  void set_mdot(ko::scs_scalar_view<p> mdot);

  void set_gamma(double);
  void compute_diagonal();
private:
  TpetraLinearSystem& linsys_;

  elem_entity_view_t<p> entities_;
  ko::scalar_view<p> volume_;
  ko::scs_vector_view<p> mapped_area_;
  ko::scs_scalar_view<p> mdot_;

  double gamma_{1};
};

} // namespace nalu
} // namespace Sierra

#endif

