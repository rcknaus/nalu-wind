#ifndef ExecutePolyFunction_h
#define ExecutePolyFunction_h

#include "MatrixFreeTraits.h"
#include "stk_util/util/ReportHandler.hpp"

#include <type_traits>
#include <utility>
#include <string>

namespace sierra {
namespace nalu {

#define MAKE_INVOKEABLE_P(func) \
  class func##_##invokeable \
  { \
  public: \
    template <int p, typename... Args>  static void invoke(Args&&... args) { func<p>(std::forward<Args>(args)...); } \
  }

template <class Invokeable, typename... Args>
void execute_poly_function(int p, Args&&... args)
{
  switch (p)
  {
    case POLY1: return Invokeable::template invoke<POLY1>(std::forward<Args>(args)...);
    case POLY2: return Invokeable::template invoke<POLY2>(std::forward<Args>(args)...);
    case POLY3: return Invokeable::template invoke<POLY3>(std::forward<Args>(args)...);
    case POLY4: return Invokeable::template invoke<POLY4>(std::forward<Args>(args)...);
    default: {
      ThrowRequireMsg(false, "invalid order: " + std::to_string(p));
      return Invokeable::template invoke<POLY1>(std::forward<Args>(args)...);
    }
  }
}


} // namespace nalu
} // namespace Sierra

#endif
