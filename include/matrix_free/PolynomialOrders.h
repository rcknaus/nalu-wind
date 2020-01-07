#ifndef POLYNOMIAL_ORDER_H
#define POLYNOMIAL_ORDER_H

#include <type_traits>

#ifndef NALU_POLYNOMIAL_ORDER1
#define NALU_POLYNOMIAL_ORDER1 1
#define NALU_NODES1 2
#endif

#ifndef NALU_POLYNOMIAL_ORDER2
#define NALU_POLYNOMIAL_ORDER2 2
#define NALU_NODES2 3
#endif

#ifndef NALU_POLYNOMIAL_ORDER3
#define NALU_POLYNOMIAL_ORDER3 3
#define NALU_NODES3 4
#endif

#ifndef NALU_POLYNOMIAL_ORDER4
#define NALU_POLYNOMIAL_ORDER4 4
#define NALU_NODES4 5
#endif

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace inst {
enum {
  P1 = NALU_POLYNOMIAL_ORDER1,
  N1 = NALU_NODES1,
  P2 = NALU_POLYNOMIAL_ORDER2,
  N2 = NALU_NODES2,
  P3 = NALU_POLYNOMIAL_ORDER3,
  N3 = NALU_NODES3,
  P4 = NALU_POLYNOMIAL_ORDER4,
  N4 = NALU_NODES4
};
}

#define INSTANTIATE_TYPE(type, Name)                                           \
  template type Name<inst::P1>;                                                \
  template type Name<inst::P2>;                                                \
  template type Name<inst::P3>;                                                \
  template type Name<inst::P4>

#define INSTANTIATE_POLYCLASS(ClassName) INSTANTIATE_TYPE(class, ClassName)
#define INSTANTIATE_POLYSTRUCT(ClassName) INSTANTIATE_TYPE(struct, ClassName)

#define P_INVOKEABLE(func)                                                     \
  template <int p, typename... Args>                                           \
  auto func(Args&&... args)                                                    \
    ->decltype(impl::func##_t<p>::invoke(std::forward<Args>(args)...))         \
  {                                                                            \
    return impl::func##_t<p>::invoke(std::forward<Args>(args)...);             \
  }

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif