/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CVFEMTypeDefs_h
#define CVFEMTypeDefs_h

#include <LocalArray.h>
#include <ScratchWorkView.h>
#include <SimdInterface.h>
#include <KokkosInterface.h>

#include <array>
#include <type_traits>

namespace sierra { namespace nalu { namespace ko {
template <typename ExecSpace> struct ViewTraits {};

template <> struct ViewTraits<Kokkos::DefaultHostExecutionSpace>
{
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Aligned | Kokkos::Restrict>;
  using memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
  using layout = Kokkos::LayoutRight;
};

enum class FieldType
{
  NODAL_SCALAR, NODAL_VECTOR, SCS_SCALAR, SCS_VECTOR
};

template <int p, FieldType, typename ExecSpace> struct GlobalArrayTypeSelector {};

template <int p, typename ExecSpace> struct GlobalArrayTypeSelector<p, FieldType::NODAL_SCALAR, ExecSpace>
{
  static constexpr int n1D = p + 1;
  using type = Kokkos::View<DoubleType*[p + 1][p + 1][p + 1],
      typename ViewTraits<ExecSpace>::layout,
      typename ViewTraits<ExecSpace>::memory_space,
      typename ViewTraits<ExecSpace>::memory_traits>;
};

template <int p, typename ExecSpace> struct GlobalArrayTypeSelector<p, FieldType::NODAL_VECTOR, ExecSpace>
{
  static constexpr int n1D = p + 1;
  using type = Kokkos::View<DoubleType*[n1D][n1D][n1D][3],
      typename ViewTraits<ExecSpace>::layout,
      typename ViewTraits<ExecSpace>::memory_space,
      typename ViewTraits<ExecSpace>::memory_traits>;
};

template <int p, typename ExecSpace> struct GlobalArrayTypeSelector<p, FieldType::SCS_SCALAR, ExecSpace>
{
  static constexpr int n1D = p + 1;

  using type = Kokkos::View<DoubleType*[3][n1D][n1D][n1D],
      typename ViewTraits<ExecSpace>::layout,
      typename ViewTraits<ExecSpace>::memory_space,
      typename ViewTraits<ExecSpace>::memory_traits>;
};

template <int p, typename ExecSpace> struct GlobalArrayTypeSelector<p, FieldType::SCS_VECTOR, ExecSpace>
{
  static constexpr int n1D = p + 1;

  using type = Kokkos::View<DoubleType*[3][n1D][n1D][n1D][3],
      typename ViewTraits<ExecSpace>::layout,
      typename ViewTraits<ExecSpace>::memory_space,
      typename ViewTraits<ExecSpace>::memory_traits>;
};

template <int p, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using scalar_view = typename GlobalArrayTypeSelector<p, FieldType::NODAL_SCALAR, ExecSpace>::type;
template <int p, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using vector_view = typename GlobalArrayTypeSelector<p, FieldType::NODAL_VECTOR, ExecSpace>::type;
template <int p, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using scs_scalar_view = typename GlobalArrayTypeSelector<p, FieldType::SCS_SCALAR, ExecSpace>::type;
template <int p, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using scs_vector_view = typename GlobalArrayTypeSelector<p, FieldType::SCS_VECTOR, ExecSpace>::type;

}}}

namespace sierra { namespace nalu {

using node_offset_view_type = Kokkos::View<int*>;

// struct is necessary as a work-around for intel 19
template <int p> struct elem_ordinal_view
{
  static constexpr int n1D = p + 1;
  using type = Kokkos::View<int*[simdLen][n1D][n1D][n1D]>;
};
template <int p> using elem_ordinal_view_t = typename elem_ordinal_view<p>::type;

template <int p> struct elem_entity_view
{
  static constexpr int n1D = p + 1;
  using type = Kokkos::View<stk::mesh::Entity*[simdLen][n1D][n1D][n1D]>;
};
template <int p> using elem_entity_view_t = typename elem_entity_view<p>::type;


template <int p, typename Scalar = DoubleType> struct CVFEMViews {
  static constexpr int n1D = p + 1;
  static constexpr int nscs = p;
  static constexpr int dim = 3;
  static constexpr int npe = n1D*n1D*n1D;

  using nodal_scalar_array = LocalArray<Scalar[n1D][n1D][n1D]>;
  using nodal_scalar_view = la::view_type<nodal_scalar_array>;
  using nodal_scalar_wsv = ScratchWorkView<nodal_scalar_array::size, nodal_scalar_view>;

  using nodal_vector_array = LocalArray<Scalar[n1D][n1D][n1D][dim]>;
  using nodal_vector_view = la::view_type<nodal_vector_array>;
  using nodal_vector_wsv = ScratchWorkView<nodal_vector_array::size, nodal_vector_view>;

  using nodal_tensor_array = LocalArray<Scalar[n1D][n1D][n1D][dim][dim]>;
  using nodal_tensor_view = la::view_type<nodal_tensor_array>;
  using nodal_tensor_wsv = ScratchWorkView<nodal_tensor_array::size, nodal_tensor_view>;

  using scs_scalar_array = LocalArray<Scalar[dim][n1D][n1D][n1D]>;
  using scs_scalar_view = la::view_type<scs_scalar_array>;
  using scs_scalar_wsv = ScratchWorkView<scs_scalar_array::size, scs_scalar_view>;

  using scs_vector_array =  LocalArray<Scalar[dim][n1D][n1D][n1D][dim]>;
  using scs_vector_view = la::view_type<scs_vector_array>;
  using scs_vector_wsv = ScratchWorkView<scs_vector_array::size, scs_vector_view>;

  using matrix_array = LocalArray<Scalar[npe][npe]>;
  using matrix_view = la::view_type<matrix_array>;

  using matrix_vector_array = LocalArray<Scalar[npe*dim][npe*dim]>;
  using matrix_vector_view = la::view_type<matrix_vector_array>;

  using nodal_coefficient_matrix = LocalArray<Scalar[p+1][p+1]>;



};
template <typename Scalar, int p> using nodal_scalar_array = typename CVFEMViews<p,Scalar>::nodal_scalar_array;
template <typename Scalar, int p> using nodal_vector_array = typename CVFEMViews<p,Scalar>::nodal_vector_array;
template <typename Scalar, int p> using nodal_tensor_array = typename CVFEMViews<p,Scalar>::nodal_tensor_array;
template <typename Scalar, int p> using scs_scalar_array = typename CVFEMViews<p,Scalar>::scs_scalar_array;
template <typename Scalar, int p> using scs_vector_array = typename CVFEMViews<p,Scalar>::scs_vector_array;
template <typename Scalar, int p> using nodal_coefficient_matrix = LocalArray<double[p+1][p+1]>;
template <typename Scalar, int p> using linear_nodal_coefficient_matrix = LocalArray<double[2][p+1]>;
template <typename Scalar, int p> using linear_scs_coefficient_matrix = LocalArray<double[2][p]>;
template <typename Scalar, int p> using linear_nodal_matrix_array = LocalArray<double[2][p+1]>;
template <typename Scalar, int p> using linear_scs_matrix_array = LocalArray<double[2][p]>;
template <typename Scalar, int p> using scs_matrix_array = LocalArray<double[p+1][p+1]>;
template <typename Scalar, int p> using nodal_matrix_array = LocalArray<double[p+1][p+1]>;
template <typename Scalar, int p> using matrix_array = typename CVFEMViews<p,Scalar>::matrix_array;


template <int p, typename Scalar = DoubleType>
using nodal_scalar_view = typename CVFEMViews<p,Scalar>::nodal_scalar_view;

template <int p, typename Scalar = DoubleType>
using nodal_scalar_workview = typename CVFEMViews<p,Scalar>::nodal_scalar_wsv;

template <int p, typename Scalar = DoubleType>
using nodal_vector_view = typename CVFEMViews<p,Scalar>::nodal_vector_view;

template <int p, typename Scalar = DoubleType>
using nodal_vector_workview = typename CVFEMViews<p,Scalar>::nodal_vector_wsv;

template <int p, typename Scalar = DoubleType>
using nodal_tensor_view = typename CVFEMViews<p,Scalar>::nodal_tensor_view;

template <int p, typename Scalar = DoubleType>
using nodal_tensor_workview = typename CVFEMViews<p,Scalar>::nodal_tensor_wsv;

template <int p, typename Scalar = DoubleType>
using scs_scalar_view = typename CVFEMViews<p,Scalar>::scs_scalar_view;

template <int p, typename Scalar = DoubleType>
using scs_scalar_workview = typename CVFEMViews<p,Scalar>::scs_scalar_wsv;

template <int p, typename Scalar = DoubleType>
using scs_vector_view = typename CVFEMViews<p,Scalar>::scs_vector_view;

template <int p, typename Scalar = DoubleType>
using scs_vector_workview = typename CVFEMViews<p,Scalar>::scs_vector_wsv;

template <int p, typename Scalar = DoubleType>
using matrix_view = typename CVFEMViews<p, Scalar>::matrix_view;

template <int p, typename Scalar = DoubleType>
using matrix_vector_view = typename CVFEMViews<p, Scalar>::matrix_vector_view;

template <int p, typename Scalar = DoubleType>
using nodal_matrix_view = nodal_coefficient_matrix<Scalar, p>;

template <int p, typename Scalar = DoubleType>
using scs_matrix_view = nodal_coefficient_matrix<Scalar, p>;

template <int p, typename Scalar = DoubleType>
using linear_nodal_matrix_view = linear_nodal_coefficient_matrix<Scalar, p>;

template <int p, typename Scalar = DoubleType>
using linear_scs_matrix_view = linear_scs_coefficient_matrix<Scalar, p>;

using node_map_view = Kokkos::View<int*>;

#define CVFEMTypeDefsDim(x,y,z) \
  using y##_##z##_##view = typename x::y##_##z##_##view; \
  using y##_##z##_##workview = typename x::y##_##z##_##wsv;

#define DeclareCVFEMTypeDefs(x) \
  CVFEMTypeDefsDim(x,nodal,scalar) \
  CVFEMTypeDefsDim(x,nodal,vector) \
  CVFEMTypeDefsDim(x,nodal,tensor) \
  CVFEMTypeDefsDim(x,scs,scalar) \
  CVFEMTypeDefsDim(x,scs,vector) \
  using matrix##_##view = typename x::matrix##_##view; \
  using matrix##_##vector##_##view = typename x::matrix##_##vector##_##view

} // namespace nalu
} // namespace Sierra

#endif
