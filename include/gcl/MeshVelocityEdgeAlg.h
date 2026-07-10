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
};

template <>
struct SubInterp<AlgTraitsHex8>
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

/*

TET SUB FACE

1: 5 8 15 14
2: 8 15 11 6
3: 7 13 15 8
4: 12 14 15 13
5: 14 10 11 15
6: 11 9 13 15


vol

1: 1 5 8 7 12 14 15 13
2: 2 6 8 5 10 11 15 14
3: 3 7 8 6 9 14 15 11
4: 4 10 14 12 9 11 15 13

*/

template <>
struct SubInterp<AlgTraitsTet4>
{
  using traits_t = AlgTraitsTet4;
  using basis_t = Tet4Basis;
  static constexpr auto interp =
    utils::interpolants<basis_t>(ArrayND<double[1][3]>{{
    }});

  static constexpr ArrayND<int[12][4]> scs_face_node_map = {};
};

/*

PYRAMID

pyr subface

1: 6 10 19 13
2: 7 10 19 15
3: 8 10 19 17
4: 9 18 19 10
5: 12 13 19 18
6: 11 16 19 13
7: 14 17 19 15
8: 16 18 19 17


vol
1: 01 06 10 09 12 13 19 18 NA NA
2: 02 07 10 06 11 15 19 13 NA NA
3: 03 08 10 07 14 17 19 14 NA NA
4: 04 09 10 08 16 18 19 17 NA NA
5: 05 19 16 18 12 13 11 15 14 16


*/

template <>
struct SubInterp<AlgTraitsPyr5>
{
  using traits_t = AlgTraitsPyr5;
  using basis_t = Pyr5Basis;
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

/*

WED SUBFACE

 1: 7 10 21 17
 2: 8 10 21 19
 3: 11 9 20 21
 4: 11 17 21 14
 5: 14 12 19 21
 6: 13 14 21 20
 7: 16 17 21 20
 8: 17 15 18 21
 9: 20 21 19 18

 vol

1: 1 16 17 7 9 20 21 10
2: 10 7 2 8 21 17 15 19
3: 9 10 8 3 20 21 19 18
4: 20 16 17 21 13 4 11 14
5: 21 17 15 19 14 11 5 12
5: 20 21 19 18 13 14 12 6


*/

template <>
struct SubInterp<AlgTraitsWed6>
{
  using traits_t = AlgTraitsWed6;
  using basis_t = Wed6Basis;
  static constexpr auto interp =
    utils::interpolants<basis_t>(ArrayND<double[1][3]>{{

    }});

  static constexpr ArrayND<int[12][4]> scs_face_node_map = {};
};

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
