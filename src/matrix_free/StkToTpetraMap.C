#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/MakeRCP.h"

#include <Teuchos_ArrayView.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Ptr.hpp>
#include "Teuchos_RCP.hpp"
#include <Tpetra_ConfigDefs.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/Selector.hpp"

#include <algorithm>
#include <cstddef>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

enum class SolutionPointStatus {
  owned_not_shared = 1,
  shared_not_owned = 2,
  owned_and_shared = 3,
  ghosted = 4,
};

enum class SolutionPointType {
  regular = 0,
  periodic_master = 1,
  periodic_slave = 2,
};

class SolutionPointCategorizer
{
public:
  SolutionPointCategorizer(
    const ngp::Mesh& mesh_in,
    const stk::mesh::Field<stk::mesh::EntityId>& gid_field_in,
    stk::mesh::PartVector periodicParts = {})
    : mesh(mesh_in),
      gid_field(gid_field_in),
      periodic_selector(stk::mesh::selectUnion(periodicParts))
  {
  }

  SolutionPointStatus status(stk::mesh::Entity) const;
  SolutionPointType type(stk::mesh::Entity) const;

  bool owned(stk::mesh::Entity) const;
  bool shared_not_owned(stk::mesh::Entity) const;
  bool owned_or_shared(stk::mesh::Entity) const;

private:
  stk::mesh::EntityId field_global_id(stk::mesh::Entity e) const
  {
    return *stk::mesh::field_data(gid_field, e);
  }

  stk::mesh::EntityId stk_mesh_global_id(stk::mesh::Entity e) const
  {
    return mesh.identifier(e);
  }

  SolutionPointStatus regular_status(stk::mesh::Entity) const;
  SolutionPointStatus periodic_status(stk::mesh::Entity) const;
  bool is_slave(stk::mesh::Entity) const;

  const ngp::Mesh mesh;
  const stk::mesh::Field<stk::mesh::EntityId>& gid_field;
  stk::mesh::Selector periodic_selector;
};

SolutionPointStatus
SolutionPointCategorizer::regular_status(stk::mesh::Entity node) const
{
  ngp::ProfilingBlock pf("SolutionPointCategorizer::regular_status");

  const stk::mesh::Bucket& b = mesh.get_bulk_on_host().bucket(node);
  const bool entity_owned = b.owned();
  const bool entity_shared = b.shared();

  if (entity_owned && entity_shared) {
    return SolutionPointStatus::owned_and_shared;
  } else if (!entity_owned && entity_shared) {
    return SolutionPointStatus::shared_not_owned;
  } else if (entity_owned && !entity_shared) {
    return SolutionPointStatus::owned_not_shared;
  } else {
    return SolutionPointStatus::ghosted;
  }
}

bool
SolutionPointCategorizer::is_slave(stk::mesh::Entity node) const
{
  return (stk_mesh_global_id(node) != field_global_id(node));
}

bool
SolutionPointCategorizer::owned(stk::mesh::Entity e) const
{
  if (type(e) == SolutionPointType::periodic_slave)
    return false;

  const auto status = regular_status(e);
  return status == SolutionPointStatus::owned_and_shared ||
         status == SolutionPointStatus::owned_not_shared;
}

bool
SolutionPointCategorizer::shared_not_owned(stk::mesh::Entity e) const
{
  if (type(e) == SolutionPointType::periodic_slave)
    return false;

  return regular_status(e) == SolutionPointStatus::shared_not_owned;
}

bool
SolutionPointCategorizer::owned_or_shared(stk::mesh::Entity e) const
{
  if (type(e) == SolutionPointType::periodic_slave)
    return false;

  return regular_status(e) != SolutionPointStatus::ghosted;
}

SolutionPointStatus
SolutionPointCategorizer::periodic_status(stk::mesh::Entity node) const
{
  return regular_status(mesh.get_bulk_on_host().get_entity(
    stk::topology::NODE_RANK, field_global_id(node)));
}

SolutionPointType
SolutionPointCategorizer::type(stk::mesh::Entity solPoint) const
{
  ngp::ProfilingBlock pf("SolutionPointCategorizer::type");

  const auto& b = mesh.get_bulk_on_host().bucket(solPoint);
  bool periodicSolPoint = false;
  for (const auto* part : b.supersets()) {
    if (periodic_selector(*part)) {
      periodicSolPoint = true;
    }
  }
  if (periodicSolPoint) {
    if (is_slave(solPoint)) {
      return SolutionPointType::periodic_slave;
    }
    return SolutionPointType::periodic_master;
  }
  return SolutionPointType::regular;
}

SolutionPointStatus
SolutionPointCategorizer::status(stk::mesh::Entity e) const
{
  ngp::ProfilingBlock pf("SolutionPointCategorizer::status");

  switch (type(e)) {
  case SolutionPointType::periodic_master: {
    return periodic_status(e);
  }
  case SolutionPointType::periodic_slave: {
    return periodic_status(e);
  }
  default: {
    return regular_status(e);
  }
  }
}

template <typename Categorizer, typename Func>
void
for_category(
  const ngp::Mesh& mesh,
  const stk::mesh::Selector& active,
  Categorizer cat,
  Func func)
{
  ngp::ProfilingBlock pf("for_category");

  const auto& bulk = mesh.get_bulk_on_host();
  const auto buckets = bulk.get_buckets(stk::topology::NODE_RANK, active);
  for (const auto* ib : buckets) {
    for (const auto node : *ib) {
      if (cat(node)) {
        func(node);
      }
    }
  }
}

template <typename Categorizer>
int
count_category(
  const ngp::Mesh& mesh, const stk::mesh::Selector& active, Categorizer cat)
{
  ngp::ProfilingBlock pf("count_category");

  int ent_count = 0;
  for_category(
    mesh, active, cat, [&ent_count](stk::mesh::Entity) { ++ent_count; });
  return ent_count;
}

int
compute_max_owned_row_id(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& gid_field,
  const stk::mesh::Selector& active,
  stk::mesh::PartVector periodicParts)
{
  ngp::ProfilingBlock pf("compute_max_owned_row_id");

  SolutionPointCategorizer solutionCat(mesh, gid_field, periodicParts);
  const auto num_owned_rows =
    count_category(mesh, active, [&solutionCat](stk::mesh::Entity e) {
      return solutionCat.owned(e);
    });
  return num_owned_rows;
}

template <typename T, typename Category>
std::vector<stk::mesh::Entity>
node_entities(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<T>& gid_field,
  const stk::mesh::Selector& active,
  Category cat)
{
  ngp::ProfilingBlock pf("node_entities");


  std::vector<stk::mesh::Entity> ents;
  ents.reserve(count_category(mesh, active, cat));
  for_category(
    mesh, active, cat, [&](stk::mesh::Entity e) { ents.push_back(e); });

  std::sort(
    ents.begin(), ents.end(), [&](stk::mesh::Entity a, stk::mesh::Entity b) {
      return *stk::mesh::field_data(gid_field, a) <
             *stk::mesh::field_data(gid_field, b);
    });
  auto iter = std::unique(
    ents.begin(), ents.end(), [&](stk::mesh::Entity a, stk::mesh::Entity b) {
      return *stk::mesh::field_data(gid_field, a) ==
             *stk::mesh::field_data(gid_field, b);
    });
  ents.erase(iter, ents.end());
  return ents;
}

} // namespace

Teuchos::RCP<const map_type>
owned_row_map(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& gid_field,
  const stk::mesh::Selector& active,
  stk::mesh::PartVector periodic_parts)
{
  ngp::ProfilingBlock pf("owned_row_map");
  return make_rcp<map_type>(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    compute_max_owned_row_id(mesh, gid_field, active, periodic_parts), 1,
    make_rcp<Teuchos::MpiComm<int>>(mesh.get_bulk_on_host().parallel()));
}

size_t
get_global_min_index(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& gid_field,
  const stk::mesh::Selector& active,
  stk::mesh::PartVector periodicParts)
{
  ngp::ProfilingBlock pf("get_global_min_index");
  return owned_row_map(mesh, gid_field, active, periodicParts)
    ->getMinGlobalIndex();
}

void
fill_id_fields(
  const ngp::Mesh& mesh,
  const stk::mesh::Selector& active,
  stk::mesh::Field<stk::mesh::EntityId>& stk_gid_field,
  stk::mesh::Field<typename map_type::global_ordinal_type>& tpetra_gid_field,
  stk::mesh::PartVector periodicParts)
{
  SolutionPointCategorizer solutionCat(mesh, stk_gid_field, periodicParts);
  auto owned_category = [&solutionCat](stk::mesh::Entity entity) {
    return solutionCat.owned(entity);
  };

  for (const auto* ib :
       mesh.get_bulk_on_host().get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto e : *ib) {
      *stk::mesh::field_data(stk_gid_field, e) = mesh.identifier(e);
    }
  }

  auto owned_ents = node_entities(mesh, stk_gid_field, active, owned_category);
  const auto gmin =
    get_global_min_index(mesh, stk_gid_field, active, periodicParts);
  int g_offset = 0;
  for (auto e : owned_ents) {
    *stk::mesh::field_data(tpetra_gid_field, e) = gmin + g_offset;
    ++g_offset;
  }
  stk::mesh::copy_owned_to_shared(mesh.get_bulk_on_host(), {&tpetra_gid_field});
}

void
fill_tpetra_id_field(
  const ngp::Mesh& mesh,
  const stk::mesh::Selector& active,
  const stk::mesh::Field<stk::mesh::EntityId>& stk_gid_field,
  stk::mesh::Field<typename map_type::global_ordinal_type>& tpetra_gid_field,
  stk::mesh::PartVector periodicParts)
{
  ngp::ProfilingBlock pf("fill_tpetra_id_field");

  SolutionPointCategorizer solutionCat(mesh, stk_gid_field, periodicParts);
  auto owned_category = [&solutionCat](stk::mesh::Entity entity) {
    return solutionCat.owned(entity);
  };

  {
    ngp::ProfilingBlock pf_inner("fill_owned_node_entities");
    const auto owned_ents =
        node_entities(mesh, stk_gid_field, active, owned_category);

    const auto gmin =
        get_global_min_index(mesh, stk_gid_field, active, periodicParts);

    int g_offset = 0;
    for (auto e : owned_ents) {
      *stk::mesh::field_data(tpetra_gid_field, e) = gmin + g_offset;
      ++g_offset;
    }

    {
      ngp::ProfilingBlock pf_inner_inner("copy owned to shared");
      stk::mesh::copy_owned_to_shared(mesh.get_bulk_on_host(), {&tpetra_gid_field});
    }
  }
}

Teuchos::RCP<const map_type>
owned_and_shared_row_map(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& stk_gid_field,
  const stk::mesh::Field<typename map_type::global_ordinal_type>&
    tpetra_gid_field,
  const stk::mesh::Selector& active,
  stk::mesh::PartVector periodicParts)
{
  ngp::ProfilingBlock("owned_and_shared_row_map");

  SolutionPointCategorizer solution_cat(mesh, stk_gid_field, periodicParts);

  std::vector<typename map_type::global_ordinal_type> row_ids;
  row_ids.reserve(
    count_category(mesh, active, [&solution_cat](stk::mesh::Entity entity) {
      return solution_cat.owned_or_shared(entity);
    }));

  auto ents = node_entities(
    mesh, tpetra_gid_field, active, [&solution_cat](stk::mesh::Entity entity) {
      return solution_cat.owned(entity);
    });
  for (auto e : ents) {
    row_ids.push_back(*stk::mesh::field_data(tpetra_gid_field, e));
  }
  ents = node_entities(
    mesh, tpetra_gid_field, active, [&solution_cat](stk::mesh::Entity entity) {
      return solution_cat.shared_not_owned(entity);
    });
  for (auto e : ents) {
    row_ids.push_back(*stk::mesh::field_data(tpetra_gid_field, e));
  }

  return make_rcp<map_type>(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), row_ids, 1,
    make_rcp<Teuchos::MpiComm<int>>(mesh.get_bulk_on_host().parallel()));
}

std::unordered_map<stk::mesh::EntityId, int>
global_to_local_id_map(
  const ngp::Mesh& mesh,
  const stk::mesh::Field<stk::mesh::EntityId>& stk_gid_field,
  const stk::mesh::Field<typename map_type::global_ordinal_type>&
    tpetra_gid_field,
  const stk::mesh::Selector& active,
  stk::mesh::PartVector periodicParts)
{
  ngp::ProfilingBlock("global_to_local_id_map");

  SolutionPointCategorizer solutionCat(mesh, stk_gid_field, periodicParts);

  std::unordered_map<stk::mesh::EntityId, int> g2l;
  g2l.reserve(
    count_category(mesh, active, [&solutionCat](stk::mesh::Entity entity) {
      return solutionCat.owned_or_shared(entity);
    }));

  int lid = 0;

  auto ents = node_entities(
    mesh, tpetra_gid_field, active, [&solutionCat](stk::mesh::Entity entity) {
      return solutionCat.owned(entity);
    });
  for (auto e : ents) {
    g2l.emplace(*stk::mesh::field_data(stk_gid_field, e), lid++);
  }
  ents = node_entities(
    mesh, tpetra_gid_field, active, [&solutionCat](stk::mesh::Entity entity) {
      return solutionCat.shared_not_owned(entity);
    });
  for (auto e : ents) {
    g2l.emplace(*stk::mesh::field_data(stk_gid_field, e), lid++);
  }
  return g2l;
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
