/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ElemToNodeTransfer_h
#define ElemToNodeTransfer_h

#include <stk_transfer/TransferBase.hpp>
#include <memory>

namespace stk { namespace mesh { class BulkData; } }
namespace stk { namespace mesh { class Selector; } }

namespace sierra{ namespace nalu { namespace transfer {

  void transfer_all(
    stk::mesh::BulkData& bulkSrc,
    const stk::mesh::Selector& elemSelector,
    stk::mesh::BulkData& bulkDest,
    const stk::mesh::Selector& nodeSelector);


  std::unique_ptr<stk::transfer::TransferBase> create_element_to_node_transfer(
    stk::mesh::BulkData& bulkSrc,
    const stk::mesh::Selector& elemSelector,
    stk::mesh::BulkData& bulkDest,
    const stk::mesh::Selector& nodeSelector);

}}}

#endif
