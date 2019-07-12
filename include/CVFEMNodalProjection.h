#ifndef CVFEMMNodalProjection_H
#define CVFEMMNodalProjection_H

#include "CVFEMTypeDefs.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"

#include "FieldTypeDef.h"

namespace sierra { namespace nalu{

template <int p> void nodal_average(
  elem_entity_view_t<p> elemNodes,
  elem_view::scalar_view<p> local_vol,
  elem_view::scalar_view<p> dnv,
  elem_view::scalar_view<p> q,
  ScalarFieldType& qField);

void nodal_average_field(
  int p,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  ScalarFieldType& qField);

void nodal_average_gradient(
  int p,
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& selector,
  const VectorFieldType& coordField,
  const VectorFieldType& qField,
  GenericFieldType& dqdxField);

}}

#endif

