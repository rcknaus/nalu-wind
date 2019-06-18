/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef LocalArray_h
#define LocalArray_h

#include <KokkosInterface.h>
#include <SimdInterface.h>

#include <type_traits>

namespace sierra {
namespace nalu {

// Multidimensional Kokkos::Array, basically
template <typename ArrayType>
struct alignas(KOKKOS_MEMORY_ALIGNMENT) LocalArray
{
public:
  using array_type = ArrayType;
  using value_type = typename std::remove_all_extents<ArrayType>::type;
  static constexpr int rank = std::rank<ArrayType>::value;
  static constexpr int size = sizeof(array_type) / sizeof(value_type);

  static constexpr unsigned extent(int n) { return
    (n == 0) ? std::extent<array_type, 0>::value :
    (n == 1) ? std::extent<array_type, 1>::value :
    (n == 2) ? std::extent<array_type, 2>::value :
    (n == 3) ? std::extent<array_type, 3>::value :
    (n == 4) ? std::extent<array_type, 4>::value :
    (n == 5) ? std::extent<array_type, 5>::value :
               std::extent<array_type, 6>::value;
  }

  static constexpr int extent_int(int n) { return extent(n); }

  array_type internal_data_;
  KOKKOS_FORCEINLINE_FUNCTION value_type* data() { return (value_type*)internal_data_; }
  KOKKOS_FORCEINLINE_FUNCTION const value_type* data() const { return (const value_type*)internal_data_; }
private:
  template <int dim_arg, typename T> using enable_dim_t = typename std::enable_if<rank == dim_arg, T>::type;
public:
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<1, T>& operator[](int i) { return internal_data_[i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<1, T> operator[](int i) const { return internal_data_[i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<1, T>& operator()(int i) { return internal_data_[i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<1, T> operator()(int i) const { return internal_data_[i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<2, T>& operator()(int j, int i) { return internal_data_[j][i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<2, T> operator()(int j, int i) const { return internal_data_[j][i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<3, T>& operator()(int k, int j, int i) { return internal_data_[k][j][i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<3, T> operator()(int k, int j, int i) const { return internal_data_[k][j][i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<4, T>& operator()(int l, int k, int j, int i) { return internal_data_[l][k][j][i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<4, T> operator()(int l, int k, int j, int i) const { return internal_data_[l][k][j][i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<5, T>& operator()(int m, int l, int k, int j, int i) { return internal_data_[m][l][k][j][i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<5, T> operator()(int m, int l, int k, int j, int i) const  { return internal_data_[m][l][k][j][i]; }

  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  enable_dim_t<6, T>& operator()(int n, int m, int l, int k, int j, int i) { return internal_data_[n][m][l][k][j][i]; }
  template <typename T = value_type> KOKKOS_FORCEINLINE_FUNCTION
  constexpr enable_dim_t<6, T> operator()(int n, int m, int l, int k, int j, int i) const  { return internal_data_[n][m][l][k][j][i]; }
};


//template <typename ArrayType, typename = void> struct LocalData{};
//
//template <typename ArrayType>
//struct LocalData<ArrayType, typename std::enable_if<std::rank<ArrayType>::value == 1>::type>
//{
//  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
//  using value_type = typename std::remove_all_extents<ArrayType>::type;
//  value_type internal_data_[extent_0];
//
//  constexpr unsigned extent(int) { return extent_0; }
//  constexpr int extent_int(int n) { return extent(n); };
//
//  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator()(int i) const { return internal_data_[i]; }
//  KOKKOS_FORCEINLINE_FUNCTION value_type& operator()(int i) { return internal_data_[i]; }
//};
//
//template <typename ArrayType>
//struct LocalData<ArrayType, typename std::enable_if<std::rank<ArrayType>::value == 2>::type>
//{
//  static constexpr int extent_0 = std::extent<ArrayType, 0>::value;
//  static constexpr int extent_1 = std::extent<ArrayType, 1>::value;
//  using value_type = typename std::remove_all_extents<ArrayType>::type;
//  value_type internal_data_[extent_1][extent_0];
//
//  constexpr unsigned extent(int n) { return ( n== 0) ? extent_0 : extent_1;}
//  constexpr int extent_int(int n) { return extent(n); };
//
//  KOKKOS_FORCEINLINE_FUNCTION constexpr value_type operator()(int j, int i) const { return internal_data_[j][i]; }
//  KOKKOS_FORCEINLINE_FUNCTION value_type& operator()(int j, int i) { return internal_data_[j][i]; }
//};



namespace la {
template <typename ExecSpace> struct ViewTraits{};

template <> struct ViewTraits<Kokkos::DefaultExecutionSpace> {
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict | Kokkos::Aligned >;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using layout = Kokkos::LayoutRight;
};

template <typename LocalArray, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using view_type = Kokkos::View<
    typename LocalArray::array_type,
    typename ViewTraits<ExecSpace>::layout,
    typename ViewTraits<ExecSpace>::memory_space,
    typename ViewTraits<ExecSpace>::memory_traits
    >;

template <typename LocalArray, typename ExecSpace = Kokkos::DefaultExecutionSpace>
using const_view_type = Kokkos::View<
    typename LocalArray::array_type,
    typename ViewTraits<ExecSpace>::layout,
    typename ViewTraits<ExecSpace>::memory_space,
    typename ViewTraits<ExecSpace>::memory_traits
    >;

template <typename LocalArrayType> view_type<LocalArrayType> make_view(LocalArrayType& la)
    { return view_type<LocalArrayType>(la.data());}
template <typename LocalArrayType> const_view_type<LocalArrayType> make_view(const LocalArrayType& la)
    { return const_view_type<LocalArrayType>(la.data());}
template <typename LocalArrayType> const_view_type<LocalArrayType> make_const_view(const LocalArrayType& la)
    { return const_view_type<LocalArrayType>(la.data());}

template <typename LocalArrayType> void fill(LocalArrayType& la, const typename LocalArrayType::value_type& val)
{
  auto* KOKKOS_RESTRICT data_ptr = la.data();
  for (int k = 0; k < LocalArrayType::size; ++k) data_ptr[k] = val;
}

template <typename LocalArrayType> void zero(LocalArrayType& la) { fill(la, (typename LocalArrayType::value_type)(0)); }

template <
typename LocalArrayType,
typename Enable = typename std::enable_if<!std::is_array<LocalArrayType>::value>::type>
LocalArrayType zero()
{
  LocalArrayType la;
  zero(la);
  return la;
}

template <
typename ArrayType,
typename Enable = typename std::enable_if<std::is_array<ArrayType>::value>::type>
LocalArray<ArrayType> zero()
{
  LocalArray<ArrayType> la;
  zero(la);
  return la;
}

}

} // namespace nalu
} // namespace Sierra

#endif
