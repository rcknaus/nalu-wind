// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gcl/MeshVelocityEdgeAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "master_element/Hex8GeometryFunctions.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include <cmath>

namespace sierra {
namespace nalu {

// template <typename AlgTraits>
// void
// oriented_scatter(const stk::mesh::NgpMesh& mesh, int n_valid_elems)
// {
//   for (int si = 0; si < n_valid_elems; ++si) {
//     const auto edges = ngpMesh.get_edges(
//       stk::topology::ELEM_RANK,
//       ngpMesh.fast_mesh_index(edata.elemInfo[si].entity));

//     const int nedge = scsIpEdgeMap[ip];
//     const int iLn = lrscv[2 * ip];
//     const auto edgeID = ngpMesh.fast_mesh_index(edges[nedge]);
//     const auto edge_nodes = ngpMesh.get_nodes(stk::topology::EDGE_RANK,
//     edgeID); const auto lnElemId = edata.elemInfo[si].entityNodes[iLn]; const
//     auto lnEdgeId = edge_nodes[0];

//     const double sign = (lnElemId == lnEdgeId) ? 1.0 : -1.0;

//     Kokkos::atomic_add(
//       &edgeSweptVol.get(edgeID, 0),
//       stk::simd::get_data(swept_volume, si) * sign);
//     Kokkos::atomic_add(
//       &edgeFaceVelMag.get(edgeID, 0),
//       stk::simd::get_data(projected_volume_change, si) * sign);
//   }
// }

template <typename AlgTraits>
MeshVelocityEdgeAlg<AlgTraits>::MeshVelocityEdgeAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    elemData_(realm.meta_data()),
    modelCoords_(get_field_ordinal(realm.meta_data(), "coordinates")),
    currentCoords_(get_field_ordinal(realm.meta_data(), "current_coordinates")),
    meshDispNp1_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateNP1)),
    meshDispN_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateN)),
    edgeFaceVelMag_(get_field_ordinal(
      realm.meta_data(), "edge_face_velocity_mag", stk::topology::EDGE_RANK)),
    edgeSweptVolumeNp1_(get_field_ordinal(
      realm.meta_data(),
      "edge_swept_face_volume",
      stk::mesh::StateNP1,
      stk::topology::EDGE_RANK)),
    edgeSweptVolumeN_(get_field_ordinal(
      realm.meta_data(),
      "edge_swept_face_volume",
      stk::mesh::StateN,
      stk::topology::EDGE_RANK)),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{

  elemData_.add_cvfem_surface_me(meSCS_);

  elemData_.add_coordinates_field(
    modelCoords_, AlgTraits::nDim_, MODEL_COORDINATES);
  elemData_.add_coordinates_field(
    currentCoords_, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(meshDispNp1_, AlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(meshDispN_, AlgTraits::nDim_);
  elemData_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);

  if (!std::is_same<AlgTraits, AlgTraitsHex8>::value) {
    throw std::runtime_error("MeshVelocityEdgeAlg is only supported for Hex8");
  }
}

template <typename AlgTraits>
void
MeshVelocityEdgeAlg<AlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const DoubleType dt = realm_.get_time_step();
  const DoubleType gamma1 = realm_.get_gamma1();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto edgeFaceVelMag = fieldMgr.template get_field<double>(edgeFaceVelMag_);
  auto edgeSweptVol = fieldMgr.template get_field<double>(edgeSweptVolumeNp1_);

  const auto modelCoordsID = modelCoords_;
  const auto meshDispNp1ID = meshDispNp1_;
  const auto meshDispNID = meshDispN_;
  MasterElement* meSCS = meSCS_;

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const auto nodesPerElement = AlgTraits::nodesPerElement_;
  const auto numScsIp = AlgTraits::numScsIp_;

  edgeSweptVol.sync_to_device();
  edgeFaceVelMag.sync_to_device();

  const std::string algName =
    "compute_mesh_vel_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, stk::topology::ELEM_RANK, elemData_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* lrscv = meSCS->adjacentNodes();
      const int* scsIpEdgeMap = meSCS->scsIpEdgeOrd();

      auto& scrView = edata.simdScrView;

      const auto& mCoords = scrView.get_scratch_view_2D(modelCoordsID);
      const auto& dispNp1 = scrView.get_scratch_view_2D(meshDispNp1ID);
      const auto& dispN = scrView.get_scratch_view_2D(meshDispNID);

      static constexpr auto interp = SubInterp<AlgTraits>::interp;
      static constexpr auto ninterp =
        decltype(SubInterp<AlgTraits>::interp)::extent_int(0);
      static constexpr auto dim =
        decltype(SubInterp<AlgTraits>::interp)::extent_int(1);

      auto scs_coords_np0 = nd_zero<DoubleType[ninterp][dim]>();
      auto scs_coords_np1 = nd_zero<DoubleType[ninterp][dim]>();

      for (int i = 0; i < ninterp; i++) {
        for (int k = 0; k < nodesPerElement; k++) {
          const auto r = interp(i, k);
          for (int j = 0; j < dim; j++) {
            scs_coords_np0(i, j) += r * (mCoords(k, j) + dispN(k, j));
            scs_coords_np1(i, j) += r * (mCoords(k, j) + dispNp1(k, j));
          }
        }
      }

      for (int ip = 0; ip < numScsIp; ++ip) {
        static constexpr auto face_node =
          SubInterp<AlgTraits>::scs_face_node_map;

        ArrayND<DoubleType[8][3]> scs_vol_coords{};

        for (int n = 0; n < 4; ++n) {
          for (int d = 0; d < dim; ++d) {
            scs_vol_coords(n, d) = scs_coords_np0(face_node(ip, n), d);
          }
        }
        for (int n = 0; n < 4; ++n) {
          for (int d = 0; d < dim; ++d) {
            scs_vol_coords(n + 4, d) = scs_coords_np1(face_node(ip, n), d);
          }
        }

        const auto swept_volume = hex_volume_grandy(scs_vol_coords);
        const auto projected_timestep = gamma1 / dt;
        const auto projected_volume_change = swept_volume * projected_timestep;

        for (int si = 0; si < edata.numSimdElems; ++si) {
          const auto edges = ngpMesh.get_edges(
            stk::topology::ELEM_RANK,
            ngpMesh.fast_mesh_index(edata.elemInfo[si].entity));

          const int nedge = scsIpEdgeMap[ip];
          const int iLn = lrscv[2 * ip];
          const auto edgeID = ngpMesh.fast_mesh_index(edges[nedge]);
          const auto edge_nodes =
            ngpMesh.get_nodes(stk::topology::EDGE_RANK, edgeID);
          const auto lnElemId = edata.elemInfo[si].entityNodes[iLn];
          const auto lnEdgeId = edge_nodes[0];

          const double sign = (lnElemId == lnEdgeId) ? 1.0 : -1.0;

          Kokkos::atomic_add(
            &edgeSweptVol.get(edgeID, 0),
            stk::simd::get_data(swept_volume, si) * sign);
          Kokkos::atomic_add(
            &edgeFaceVelMag.get(edgeID, 0),
            stk::simd::get_data(projected_volume_change, si) * sign);
        }
      }
    });
  edgeSweptVol.modify_on_device();
  edgeFaceVelMag.modify_on_device();
  edgeSweptVol.sync_to_host();
  edgeFaceVelMag.sync_to_host();
}

template class MeshVelocityEdgeAlg<AlgTraitsHex8>;

#define EMPTY_DECL(TRAITS)                                                     \
  template <>                                                                  \
  MeshVelocityEdgeAlg<TRAITS>::MeshVelocityEdgeAlg(                            \
    Realm& realm, stk::mesh::Part* part)                                       \
    : Algorithm(realm, part), elemData_(realm_.meta_data())                    \
  {                                                                            \
  }                                                                            \
  template <>                                                                  \
  void MeshVelocityEdgeAlg<TRAITS>::execute()                                  \
  {                                                                            \
  }                                                                            \
  static_assert(true)

EMPTY_DECL(AlgTraitsTet4);
EMPTY_DECL(AlgTraitsWed6);
EMPTY_DECL(AlgTraitsTri3_2D);
EMPTY_DECL(AlgTraitsQuad4_2D);
EMPTY_DECL(AlgTraitsPyr5);

void
edge_to_node()
{
  nalu_ngp::run_edge_algorithm(
    "continuity residual", mesh, stk::topology::EDGE_RANK, interior,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      const auto edge = eInfo.meshIdx;
      const auto& nodes = eInfo.entityNodes;
      const auto val = edge_field(edge, 0);
      projected_field(nodes[0], 0) += val;
      projected_field(nodes[1], 0) -= val;
    });
}

void
exposed_edge_to_node()
{
  nalu_ngp::run_edge_algorithm(
    "continuity residual bc", mesh, stk::topology::EDGE_RANK, interior,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      const auto edge = eInfo.meshIdx;
      const auto& nodes = eInfo.entityNodes;
      const auto val = edge_field(edge, 0);
      projected_field(nodes[0], 0) += val;
    });
}

void
continuity_residual(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& interior,
  const stk::mesh::Selector& boundary,
  Kokkos::Array<double, 3> gammas,
  Kokkos::Array<stk::mesh::NgpField<double>, 3> rho,
  Kokkos::Array<stk::mesh::NgpField<double>, 3> vol,
  stk::mesh::NgpField<double> mdot,
  stk::mesh::NgpField<double> area_v)
{
  nalu_ngp::run_edge_algorithm(
    "continuity residual", mesh, stk::topology::EDGE_RANK, interior,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      double drho_dt = 0;
      for (int n = 0; n < 3; ++n) {
        drho_dt += gammas[n] * rho[n](mi, 0) * vol[n](mi, 0);
      }
    });
}

} // namespace nalu
} // namespace sierra
