#ifndef CONDUCTION_UPDATE_H
#define CONDUCTION_UPDATE_H

#include "matrix_free/EquationUpdate.h"
#include "matrix_free/ConductionGatheredFieldManager.h"
#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/KokkosFramework.h"
#include <stk_ngp/Ngp.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
class ConductionUpdate final : public EquationUpdate
{
public:
  ConductionUpdate(
    const stk::mesh::MetaData&,
    const ngp::Mesh&,
    const ngp::FieldManager&,
    Teuchos::ParameterList,
    stk::mesh::Selector,
    stk::mesh::Selector,
    stk::mesh::Selector);

  void initialize() final;
  void swap_states() final;
  void predict_state() final;
  void compute_preconditioner(double projected_dt) final;
  void
  compute_update(Kokkos::Array<double, 3>, ngp::Field<double>& delta) final;
  void update_solution_fields() final;
  double provide_norm() const final { return residual_norm_; };
  double provide_scaled_norm() const final { return scaled_residual_norm_; }
  void banner(std::string name, std::ostream& stream) const final;

private:
  const ngp::FieldManager& fm_;
  const ngp::Mesh& mesh_;
  const stk::mesh::Selector active_;

  ConductionSolutionUpdate<p> field_update_;
  ConductionGatheredFieldManager<p> field_gather_;

  int solution_field_ordinal_np1_{-1};
  int solution_field_ordinal_np0_{-1};
  int solution_field_ordinal_nm1_{-1};

  double initial_residual_{-1};
  double residual_norm_{0};
  double scaled_residual_norm_{0};
};

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
