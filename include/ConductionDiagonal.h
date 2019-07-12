/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ConductionDiagonal_h
#define ConductionIDiagonal_h

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

template <int p> struct ConductionInteriorDiagonal
{
  using op = conduction_element_residual<p>;

  ConductionInteriorDiagonal(
    const stk::mesh::BulkData& bulk,
    stk::mesh::Selector selector,
    TpetraLinearSystem& linsys,
    const VectorFieldType& coordField,
    const ScalarFieldType& alpha,
    const ScalarFieldType& diffusivity);

  void initialize();
  void set_gamma(double);
  void initialize_connectivity();
  void compute_diagonal();
private:
  const stk::mesh::BulkData& bulk_;
  stk::mesh::Selector selector_;
  TpetraLinearSystem& linsys_;

  elem_view::scalar_view<p> volume_;
  elem_view::scs_vector_view<p> mapped_area_;
  elem_entity_view_t<p> entities_;

  double gamma_{1};
};

} // namespace nalu
} // namespace Sierra

#endif

