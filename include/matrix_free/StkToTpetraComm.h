// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STK_TO_TPETRA_COMM_H
#define STK_TO_TPETRA_COMM_H

#include "Teuchos_RCP.hpp"
#include "stk_mesh/base/Types.hpp"

namespace Teuchos {
template <class OrdinalType>
class Comm;
}

namespace sierra {
namespace nalu {
namespace matrix_free {

Teuchos::RCP<const Teuchos::Comm<int>>
teuchos_communicator(const stk::ParallelMachine& pm);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
