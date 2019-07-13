#ifndef MFTraits_h
#define MFTraits_h

namespace sierra {
namespace nalu {

#ifndef POLY1
#define POLY1 1
#endif

#ifndef POLY2
#define POLY2 2
#endif

#ifndef POLY3
#define POLY3 3
#endif

#ifndef POLY4
#define POLY4 4
#endif

struct MF {
  static constexpr bool doMatrixFree = true;
  static constexpr int p = POLY3;
};


} // namespace nalu
} // namespace Sierra

#endif
