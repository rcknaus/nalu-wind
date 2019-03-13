/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScratchViewsHOMF_h
#define ScratchViewsHOMF_h

#include <CVFEMTypeDefs.h>
#include <ElemDataRequests.h>
#include <FieldTypeDef.h>
#include <KokkosInterface.h>
#include <SimdInterface.h>


namespace sierra{
namespace nalu{

template <int p>
class ScratchViewsHOMF
{
  DeclareCVFEMTypeDefs(CVFEMViews<p>);
  static constexpr int maxViews = 1000;
public:
  using value_type = DoubleType;

  ScratchViewsHOMF(
    const TeamHandleType& team,
    const stk::mesh::BulkData& bulkData,
    const ElemDataRequests& dataNeeded)
  {
    int numScalars = 0;
    const stk::mesh::MetaData& meta = bulkData.mesh_meta_data();
    unsigned numFields = meta.get_fields().size();

    const FieldSet& neededFields = dataNeeded.get_fields();
    for(const FieldInfo& fieldInfo : neededFields) {
      ThrowAssert(fieldInfo.field->entity_rank() == stk::topology::NODE_RANK);
      unsigned scalarsDim1 = fieldInfo.scalarsDim1;
      constexpr int n1D = p + 1;
      if ( scalarsDim1 == 1u) {
        internal_fields[fieldInfo.field->mesh_meta_data_ordinal()] = get_shmem_view_1D<DoubleType>(team, n1D* n1D* n1D);

        fieldViewsScalar[fieldInfo.field->mesh_meta_data_ordinal()] =
            nodal_scalar_view(internal_fields[fieldInfo.field->mesh_meta_data_ordinal()].data());
      }
      else {
        internal_fields[fieldInfo.field->mesh_meta_data_ordinal()] = get_shmem_view_1D<DoubleType>(team, 3*n1D* n1D* n1D);
        fieldViewsVector[fieldInfo.field->mesh_meta_data_ordinal()] =
            nodal_vector_view(internal_fields[fieldInfo.field->mesh_meta_data_ordinal()].data());
      }
      numScalars += n1D * n1D * n1D;
    }
    num_bytes_required += numScalars * sizeof(DoubleType);
  }

  value_type* get_scratch_view_ptr(const ScalarFieldType& field)
  {
    return fieldViewsScalar[field.mesh_meta_data_ordinal()].data();
  }

  value_type* get_scratch_view_ptr(const VectorFieldType& field)
  {
    return fieldViewsVector[field.mesh_meta_data_ordinal()].data();
  }
//
//  value_type* get_scratch_view_ptr(const GenericFieldType& field) // this is not safe
//  {
//    return fieldViewsTensor[field.mesh_meta_data_ordinal()].data();
//  }

  nodal_scalar_view get_scratch_view(const ScalarFieldType& field)
  {
    return nodal_scalar_view(get_scratch_view_ptr(field));
  }

  nodal_vector_view get_scratch_view(const VectorFieldType& field)
  {
    return nodal_vector_view(get_scratch_view_ptr(field));
  }
//
//  nodal_tensor_view get_scratch_view(const GenericFieldType& field)
//  {
//    return nodal_tensor_view(get_scratch_view_ptr(field));
//  }

  int total_bytes() const { return num_bytes_required; }


  std::array<const stk::mesh::Entity*, simdLen> elemNodes{{}};
  int numSimdElems{simdLen};

private:
  Kokkos::Array<SharedMemView<DoubleType*, HostShmem>, maxViews> internal_fields;
  Kokkos::Array<nodal_scalar_view, maxViews> fieldViewsScalar{{}};
  Kokkos::Array<nodal_vector_view, maxViews> fieldViewsVector{{}};
//  Kokkos::Array<SharedMemView<DoubleType*****, HostShmem>, maxViews> fieldViewsTensor{{}};

  int num_bytes_required{0};
};


} // namespace nalu
} // namespace Sierra

#endif

