/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "MatrixFreeOperator.h"

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include "element_promotion/NodeMapMaker.h"
#include "nalu_make_unique.h"

#include "BelosLinearProblem.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"

namespace sierra { namespace nalu {


} // namespace nalu
} // namespace Sierra



