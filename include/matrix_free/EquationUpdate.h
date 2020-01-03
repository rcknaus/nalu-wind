#ifndef EQUATION_UPDATE_H
#define EQUATION_UPDATE_H

#include "matrix_free/PolynomialOrders.h"
#include "Kokkos_Array.hpp"

#include "stk_ngp/Ngp.hpp"

#include <memory>

namespace sierra {
namespace nalu {
namespace matrix_free {

class EquationUpdate
{
public:
  virtual ~EquationUpdate() = default;
  virtual void initialize() = 0;
  virtual void swap_states() = 0;
  virtual void predict_state() = 0;
  virtual void compute_preconditioner(double = -1) = 0;
  virtual void
  compute_update(Kokkos::Array<double, 3>, ngp::Field<double>&) = 0;
  virtual void update_solution_fields() = 0;
  virtual double provide_norm() const = 0;
  virtual double provide_scaled_norm() const = 0;
  virtual void banner(std::string, std::ostream&) const = 0;
};

template <template <int> class PhysicsUpdate, typename... Args>
std::unique_ptr<EquationUpdate>
make_equation_update(int p, Args&&... args)
{
  switch (p) {
  case inst::P2:
    return std::unique_ptr<PhysicsUpdate<inst::P2>>(
      new PhysicsUpdate<inst::P2>(std::forward<Args>(args)...));
  case inst::P3:
    return std::unique_ptr<PhysicsUpdate<inst::P3>>(
      new PhysicsUpdate<inst::P3>(std::forward<Args>(args)...));
  case inst::P4:
    return std::unique_ptr<PhysicsUpdate<inst::P4>>(
      new PhysicsUpdate<inst::P4>(std::forward<Args>(args)...));
  default:
    return std::unique_ptr<PhysicsUpdate<inst::P1>>(
      new PhysicsUpdate<inst::P1>(std::forward<Args>(args)...));
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
