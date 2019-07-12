#ifndef SOLUTION_POINT_CATEGORIZER_H
#define SOLUTION_POINT_CATEGORIZER_H

#include <FieldTypeDef.h>
#include <Kokkos_DefaultNode.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include "stk_util/environment/Env.hpp"
#include <vector>
#include <string>

namespace sierra {
namespace nalu {

template <typename Enum> struct EnableBitMaskOperators {};

template<typename Enum> typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type
operator|(Enum lhs, Enum rhs)
{
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum> (static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}

template<typename Enum> typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type
operator&(Enum lhs, Enum rhs)
{
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum> (static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}

template<typename Enum> typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type
operator~(Enum lhs)
{
  return static_cast<Enum> (~static_cast<typename std::underlying_type<Enum>::type>(lhs));
}

#define ENABLE_BITMASK_OPERATORS(x)  \
template<>                           \
struct EnableBitMaskOperators<x>     \
{                                    \
static constexpr bool enable = true; \
};

enum class SolutionPointStatus
{
  notset = 1 << 0,
  skipped = 1 << 1,
  owned_not_shared = 1 << 2,
  shared_not_owned = 1 << 3,
  owned_and_shared = 1 << 4,
  ghosted = 1 << 5,
  nonconformal= 1 << 6
};
ENABLE_BITMASK_OPERATORS(SolutionPointStatus)

inline bool is_skipped(SolutionPointStatus status)
{
  return static_cast<bool>(status & SolutionPointStatus::skipped);
}

inline bool is_owned(SolutionPointStatus status)
{
  return static_cast<bool>((status & SolutionPointStatus::owned_not_shared) | (status & SolutionPointStatus::owned_and_shared));
}
inline bool is_shared(SolutionPointStatus status)
{
  return static_cast<bool>((status & SolutionPointStatus::shared_not_owned) | (status & SolutionPointStatus::owned_and_shared));
}
inline bool is_ghosted(SolutionPointStatus status)
{
  return static_cast<bool>(status & SolutionPointStatus::ghosted);
}

enum class SolutionPointType {
  regular = 1 << 0,
  periodic_master = 1 << 1,
  periodic_slave = 1 << 2,
  nonconformal = 1 << 3
};
ENABLE_BITMASK_OPERATORS(SolutionPointType)

#undef ENABLE_BITMASK_OPERATORS

inline bool is_regular(SolutionPointType type) {
  return static_cast<bool>(type & SolutionPointType::regular);
}

inline bool is_master(SolutionPointType type) {
  return static_cast<bool>(type & SolutionPointType::periodic_master);
}

inline bool is_slave(SolutionPointType type) {
  return static_cast<bool>(type & SolutionPointType::periodic_slave);
}

inline bool is_nonconformal(SolutionPointType type) {
  return static_cast<bool>(type & SolutionPointType::nonconformal);
}

class SolutionPointCategorizer
{
public:
  SolutionPointCategorizer(
    const stk::mesh::BulkData& bulk,
    const GlobalIdFieldType& gidField,
    stk::mesh::PartVector periodicParts = {},
    stk::mesh::PartVector nonconformalParts = {})
  : bulk_(bulk),
    globalIdField_(gidField),
    periodicSelector_(stk::mesh::selectUnion(periodicParts)),
    nonconformalSelector_(stk::mesh::selectUnion(nonconformalParts))
  {}

  SolutionPointStatus status(stk::mesh::Entity) const;
  SolutionPointType type(stk::mesh::Entity) const;
private:
  stk::mesh::EntityId nalu_mesh_global_id(stk::mesh::Entity e)  const {
    return static_cast<stk::mesh::EntityId>(*stk::mesh::field_data(globalIdField_, e));
  }

  stk::mesh::EntityId stk_mesh_global_id(stk::mesh::Entity e)  const {
    return bulk_.identifier(e);
  }

  SolutionPointStatus regular_status(stk::mesh::Entity) const;
  SolutionPointStatus special_status(stk::mesh::Entity) const;
  SolutionPointStatus periodic_status(stk::mesh::Entity) const;
  SolutionPointStatus nonconformal_status(stk::mesh::Entity) const;
  bool is_slave(stk::mesh::Entity) const;

  const stk::mesh::BulkData& bulk_;
  const GlobalIdFieldType& globalIdField_;
  stk::mesh::Selector periodicSelector_;
  stk::mesh::Selector nonconformalSelector_;
};

}
}

#endif /* SOLUTION_POINT_CATEGORIZER_H */
