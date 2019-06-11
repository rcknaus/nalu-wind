#ifndef nalu_make_unique_h
#define nalu_make_unique_h

#include <memory>
#include <Teuchos_RCP.hpp>


namespace sierra {
namespace nalu {

// replace with std::make_unique when we move to C++14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename... Args> Teuchos::RCP<T> make_rcp(Args&&... args)
{
  return Teuchos::RCP<T>(new T(std::forward<Args>(args)...));
}

}
}

#endif
