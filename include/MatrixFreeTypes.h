/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef MatrixFreeTypes_h
#define MatrixFreeTypes__h

#include "Tpetra_Vector.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Map.hpp"

namespace Teuchos { class ParameterList; }

namespace sierra { namespace nalu {
  using local_ordinal_type = int;
  using global_ordinal_type = long;
  using mv_type = Tpetra::MultiVector<double, int, long>;
  using operator_type = Tpetra::Operator<double, int, long>;
  using map_type = Tpetra::Map<int, long>;
}}

#endif
