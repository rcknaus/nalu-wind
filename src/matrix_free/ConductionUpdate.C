#include "matrix_free/ConductionUpdate.h"
#include "matrix_free/ConductionGatheredFieldManager.h"
#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/KokkosFramework.h"

#include <Teuchos_ParameterList.hpp>
#include <limits>
#include <stk_mesh/base/FieldState.hpp>
#include <stk_mesh/base/Types.hpp>

#include "Teuchos_RCP.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/MetaData.hpp"

#include "stk_ngp/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

int
get_ordinal(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state)
{
  ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return meta.get_field(stk::topology::NODE_RANK, name)
    ->field_state(state)
    ->mesh_meta_data_ordinal();
}

const stk::mesh::Field<stk::mesh::EntityId>&
get_gid_field(const stk::mesh::MetaData& meta)
{
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, conduction_info::gid_name));
  return *meta.get_field<stk::mesh::Field<stk::mesh::EntityId>>(
    stk::topology::NODE_RANK, conduction_info::gid_name);
}

const stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>&
get_tpetra_gid_field(const stk::mesh::MetaData& meta)
{
  ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, conduction_info::tpetra_gid_name));
  return *meta.get_field<
    stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>>(
    stk::topology::NODE_RANK, conduction_info::tpetra_gid_name);
}

} // namespace

template <int p>
ConductionUpdate<p>::ConductionUpdate(
  const stk::mesh::MetaData& meta,
  const ngp::Mesh& mesh_in,
  const ngp::FieldManager& fm_in,
  Teuchos::ParameterList params,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  stk::mesh::Selector flux_in)
  : fm_(fm_in),
    mesh_(mesh_in),
    active_(active_in),
    field_update_(
      params,
      mesh_in,
      get_gid_field(meta),
      get_tpetra_gid_field(meta),
      active_in,
      dirichlet_in,
      flux_in),
    field_gather_(meta, mesh_in, fm_in, active_in, dirichlet_in, flux_in),
    solution_field_ordinal_np1_(
      get_ordinal(meta, conduction_info::q_name, stk::mesh::StateNP1)),
    solution_field_ordinal_np0_(
      get_ordinal(meta, conduction_info::q_name, stk::mesh::StateN)),
    solution_field_ordinal_nm1_(
      get_ordinal(meta, conduction_info::q_name, stk::mesh::StateNM1))
{
}

template <int p>
void
ConductionUpdate<p>::initialize()
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::initialize");
  field_gather_.gather_all();
}

template <int p>
void
ConductionUpdate<p>::swap_states()
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::swap_states");
  field_gather_.swap_states();
  initial_residual_ = -1;
}

template <int p>
void
ConductionUpdate<p>::compute_preconditioner(double projected_dt)
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::compute_preconditioner");
  field_update_.compute_preconditioner(
    projected_dt, field_gather_.get_coefficient_fields());
}

namespace {

void
copy_state(
  const ngp::Mesh& mesh,
  const stk::mesh::Selector& active,
  ngp::Field<double>& dst,
  const ngp::ConstField<double>& src)
{
  ngp::ProfilingBlock pf("copy_state");

  ngp::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active,
    KOKKOS_LAMBDA(ngp::Mesh::MeshIndex mi) {
      dst.get(mi, 0) = src.get(mi, 0);
    });
  dst.modify_on_device();
}

} // namespace

template <int p>
void
ConductionUpdate<p>::predict_state()
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::predict_state");
  copy_state(
    mesh_, active_, fm_.get_field<double>(solution_field_ordinal_np1_),
    fm_.get_field<double>(solution_field_ordinal_np0_));
  field_gather_.update_solution_fields();
  initial_residual_ = -1;
  exec_space().fence();
}

template <int p>
void
ConductionUpdate<p>::compute_update(
  Kokkos::Array<double, 3> gammas, ngp::Field<double>& delta)
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::compute_update");
  field_update_.compute_residual(
    gammas, field_gather_.get_residual_fields(), field_gather_.get_bc_fields(),
    field_gather_.get_flux_fields());

  field_update_.compute_delta(
    gammas[0], field_gather_.get_coefficient_fields(), delta);

  residual_norm_ = field_update_.residual_norm();
  if (initial_residual_ < 0) {
    initial_residual_ = residual_norm_;
  }
  scaled_residual_norm_ =
    residual_norm_ /
    std::max(std::numeric_limits<double>::epsilon(), initial_residual_);
}

template <int p>
void
ConductionUpdate<p>::update_solution_fields()
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::update_solution_fields");
  field_gather_.update_solution_fields();
}

template <int p>
void
ConductionUpdate<p>::banner(std::string name, std::ostream& stream) const
{
  ngp::ProfilingBlock pf("ConductionUpdate<p>::banner");
  const int nameOffset = name.length() + 8;
  stream << std::setw(nameOffset) << std::right << name
         << std::setw(32 - nameOffset) << std::right
         << field_update_.num_iterations() << std::setw(18) << std::right
         << field_update_.final_linear_norm() << std::setw(15) << std::right
         << residual_norm_ << std::setw(14) << std::right
         << scaled_residual_norm_ << std::endl;
}
INSTANTIATE_POLYCLASS(ConductionUpdate);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
