// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MESHVELOCITYEDGEALG_H
#define MESHVELOCITYEDGEALG_H

#include "AlgTraits.h"
#include "KokkosInterface.h"
#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"
#include "ArrayND.h"

#include "master_element/CompileTimeElements.h"
#include "master_element/ElementBasis.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

template <typename AlgTraits>
struct SubInterp
{
  using traits_t = AlgTraitsHex8;
  using basis_t = Hex8Basis;
  static constexpr auto interp =
    utils::interpolants<basis_t>(ArrayND<double[19][3]>{{
      {+0, -1, -1}, // surf 1    1->2  0  8
      {+1, +0, -1}, // surf 2    2->3  1  9
      {+0, +1, -1}, // surf 3    3->4  2 10
      {-1, +0, -1}, // surf 4    1->4  3 11
      {+0, +0, -1}, //                 4 12
      {+0, -1, +1}, // surf 5    5->6  5 13
      {+1, +0, +1}, // surf 6    6->7  6 14
      {+0, +1, +1}, // surf 7    7->8  7 15
      {-1, +0, +1}, // surf 8    5->8  8 16
      {+0, +0, +1}, //                 9 17
      {+1, -1, +0}, // surf 10   2->6 10 18
      {-1, -1, +0}, // surf 9    1->5 11 19
      {+0, -1, +0}, //                12 20
      {-1, +1, +0}, // surf 12   4->8 14 21
      {+1, +1, +0}, // surf 11   3->7 13 22
      {+0, +1, +0}, //                15 23
      {+1, +0, +0}, //                16 24
      {-1, +0, +0}, //                17 25
      {+0, +0, +0}, //                18 26
    }});

  static constexpr ArrayND<int[12][4]> scs_face_node_map = {
    {{12, 0, 4, 18},
     {16, 1, 4, 18},
     {2, 4, 18, 15},
     {3, 17, 18, 4},
     {5, 12, 18, 9},
     {9, 6, 16, 18},
     {9, 7, 15, 18},
     {8, 9, 18, 17},
     {11, 12, 18, 17},
     {12, 10, 16, 18},
     {14, 15, 18, 16},
     {13, 17, 18, 15}}};
};

// template <>
// struct SubInterp<AlgTraitsHex8>
// {
//   using traits_t = AlgTraitsHex8;
//   using basis_t = Hex8Basis;
//   static constexpr auto interp =
//     utils::interpolants<basis_t>(ArrayND<double[19][3]>{{
//       {+0, -1, -1}, // surf 1    1->2  0  8
//       {+1, +0, -1}, // surf 2    2->3  1  9
//       {+0, +1, -1}, // surf 3    3->4  2 10
//       {-1, +0, -1}, // surf 4    1->4  3 11
//       {+0, +0, -1}, //                 4 12
//       {+0, -1, +1}, // surf 5    5->6  5 13
//       {+1, +0, +1}, // surf 6    6->7  6 14
//       {+0, +1, +1}, // surf 7    7->8  7 15
//       {-1, +0, +1}, // surf 8    5->8  8 16
//       {+0, +0, +1}, //                 9 17
//       {+1, -1, +0}, // surf 10   2->6 10 18
//       {-1, -1, +0}, // surf 9    1->5 11 19
//       {+0, -1, +0}, //                12 20
//       {-1, +1, +0}, // surf 12   4->8 14 21
//       {+1, +1, +0}, // surf 11   3->7 13 22
//       {+0, +1, +0}, //                15 23
//       {+1, +0, +0}, //                16 24
//       {-1, +0, +0}, //                17 25
//       {+0, +0, +0}, //                18 26
//     }});

//   static constexpr ArrayND<int[12][4]> scs_face_node_map = {
//     {{12, 0, 4, 18},
//      {16, 1, 4, 18},
//      {2, 4, 18, 15},
//      {3, 17, 18, 4},
//      {5, 12, 18, 9},
//      {9, 6, 16, 18},
//      {9, 7, 15, 18},
//      {8, 9, 18, 17},
//      {11, 12, 18, 17},
//      {12, 10, 16, 18},
//      {14, 15, 18, 16},
//      {13, 17, 18, 15}}};
// };

// template <>
// struct SubInterp<AlgTraitsTet4>
// {
//   using traits_t = AlgTraitsTet4;
//   using basis_t = Hex8Basis;
//   static constexpr auto sub_interp =
//     utils::interpolants<basis_t>(ArrayND<double[19][3]>{{
//       {+0, -1, -1}, // surf 1    1->2  0  8
//       {+1, +0, -1}, // surf 2    2->3  1  9
//       {+0, +1, -1}, // surf 3    3->4  2 10
//       {-1, +0, -1}, // surf 4    1->4  3 11
//       {+0, +0, -1}, //                 4 12
//       {+0, -1, +1}, // surf 5    5->6  5 13
//       {+1, +0, +1}, // surf 6    6->7  6 14
//       {+0, +1, +1}, // surf 7    7->8  7 15
//       {-1, +0, +1}, // surf 8    5->8  8 16
//       {+0, +0, +1}, //                 9 17
//       {+1, -1, +0}, // surf 10   2->6 10 18
//       {-1, -1, +0}, // surf 9    1->5 11 19
//       {+0, -1, +0}, //                12 20
//       {-1, +1, +0}, // surf 12   4->8 14 21
//       {+1, +1, +0}, // surf 11   3->7 13 22
//       {+0, +1, +0}, //                15 23
//       {+1, +0, +0}, //                16 24
//       {-1, +0, +0}, //                17 25
//       {+0, +0, +0}, //                18 26
//     }});

//   static constexpr ArrayND<int[12][4]> scs_face_node_map = {
//     {{12, 0, 4, 18},
//      {16, 1, 4, 18},
//      {2, 4, 18, 15},
//      {3, 17, 18, 4},
//      {5, 12, 18, 9},
//      {9, 6, 16, 18},
//      {9, 7, 15, 18},
//      {8, 9, 18, 17},
//      {11, 12, 18, 17},
//      {12, 10, 16, 18},
//      {14, 15, 18, 16},
//      {13, 17, 18, 15}}};
// };

template <typename AlgTraits>
class MeshVelocityEdgeAlg : public Algorithm
{
public:
  MeshVelocityEdgeAlg(Realm&, stk::mesh::Part*);

  virtual ~MeshVelocityEdgeAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests elemData_;

  unsigned modelCoords_{stk::mesh::InvalidOrdinal};
  unsigned currentCoords_{stk::mesh::InvalidOrdinal};
  unsigned meshDispNp1_{stk::mesh::InvalidOrdinal};
  unsigned meshDispN_{stk::mesh::InvalidOrdinal};
  unsigned edgeFaceVelMag_{stk::mesh::InvalidOrdinal};
  unsigned edgeSweptVolumeNp1_{stk::mesh::InvalidOrdinal};
  unsigned edgeSweptVolumeN_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* MESHVELOCITYEDGEALG_H */
