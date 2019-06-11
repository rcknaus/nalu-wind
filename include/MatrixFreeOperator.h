/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MatrixFreeOperator_h
#define MatrixFreeOperator_h

#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "FieldTypeDef.h"
#include "CVFEMTypeDefs.h"
#include "tpetra_linsys/TpetraMeshManager.h"
#include "element_promotion/NodeMapMaker.h"
#include "nalu_make_unique.h"

#include "Tpetra_Operator.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"


#include "MatrixFreeTypes.h"
#include "nalu_make_unique.h"

namespace Belos { template <typename Scalar, typename MV, typename OP> class LinearProblem; }

namespace sierra { namespace nalu {

template <typename InteriorOperatorType, typename BoundaryOperatorType, typename ExecSpace = Kokkos::DefaultHostExecutionSpace>
class MFOperatorParallel final : public operator_type
{
public:
  using scalar_type = double;
  using local_ordinal_type = int;
  using global_ordinal_type = long;
  using node_type = typename operator_type::node_type;
  using import_type = Tpetra::Import<local_ordinal_type, global_ordinal_type>;
  using export_type = Tpetra::Export<local_ordinal_type, global_ordinal_type>;

  MFOperatorParallel(
    InteriorOperatorType& inmfInterior,
    BoundaryOperatorType& inmfBoundary,
    global_ordinal_type maxOwnedRowLid,
    global_ordinal_type maxSharedNotOwnedRowLid,
    Teuchos::RCP<map_type> ownedMap,
    Teuchos::RCP<map_type> sharedMap,
    Teuchos::RCP<map_type> ownedAndSharedMap,
    int dim)
  : mfInterior_(inmfInterior),
    mfBoundary_(inmfBoundary),
    maxOwnedRowLid_(maxOwnedRowLid),
    maxSharedNotOwnedRowId_(maxSharedNotOwnedRowLid),
    ownedMap_(ownedMap),
    sharedMap_(sharedMap),
    ownedAndSharedMap_(ownedAndSharedMap),
    importer_(make_rcp<import_type>(ownedMap, ownedAndSharedMap)),
    exporter_(make_rcp<export_type>(sharedMap, ownedMap)),
    ownedAndSharedSln_(make_rcp<mv_type>(ownedAndSharedMap, dim)),
    sharedRHS_(make_rcp<mv_type>(sharedMap, dim)),
    ownedAndSharedSln1D_(make_rcp<mv_type>(ownedAndSharedMap, 1)),
    sharedRHS1D_(make_rcp<mv_type>(sharedMap, 1))
  {
  }

  void apply(const mv_type& ownedSolution, mv_type& ownedRHS,
    Teuchos::ETransp = Teuchos::NO_TRANS,
    double = 1.0,
    double = 0.0
    ) const final
  {
    ThrowRequire(ownedSolution.getNumVectors() == 1 || ownedSolution.getNumVectors() == 3);
    auto sln = (ownedSolution.getNumVectors() == 1) ? ownedAndSharedSln1D_ : ownedAndSharedSln_;
    auto sharedRHS = (ownedRHS.getNumVectors() == 1) ? sharedRHS1D_ : sharedRHS_;

    sln->doImport(ownedSolution, *importer_, Tpetra::INSERT);
    const auto xOwnedAndShared = sln->getLocalView<ExecSpace>();

    ownedRHS.putScalar(0.);
    auto yOwned = ownedRHS.getLocalView<ExecSpace>();

    sharedRHS->putScalar(0.);
    auto yShared = sharedRHS->getLocalView<ExecSpace>();

    mfInterior_.compute_linearized_residual(maxOwnedRowLid_, maxSharedNotOwnedRowId_, xOwnedAndShared, yOwned, yShared);
    mfBoundary_.compute_linearized_residual(maxOwnedRowLid_, maxSharedNotOwnedRowId_, xOwnedAndShared, yOwned, yShared);
    ownedRHS.doExport(*sharedRHS, *exporter_, Tpetra::ADD);
  }

  void compute_rhs(mv_type& ownedRHS) const
  {
    ownedRHS.putScalar(0.);
    sharedRHS_->putScalar(0.);
    mfInterior_.compute_rhs(maxOwnedRowLid_, maxSharedNotOwnedRowId_, ownedRHS.getLocalView<ExecSpace>(), sharedRHS_->getLocalView<ExecSpace>());
    mfBoundary_.compute_rhs(maxOwnedRowLid_, maxSharedNotOwnedRowId_, ownedRHS.getLocalView<ExecSpace>(), sharedRHS_->getLocalView<ExecSpace>());
    ownedRHS.doExport(*sharedRHS_, *exporter_, Tpetra::ADD);
  }

  Teuchos::RCP<const map_type> getDomainMap() const final { return ownedMap_; }
  Teuchos::RCP<const map_type> getRangeMap() const final { return ownedMap_; }
private:
  InteriorOperatorType& mfInterior_;
  BoundaryOperatorType& mfBoundary_;
  global_ordinal_type maxOwnedRowLid_;
  global_ordinal_type maxSharedNotOwnedRowId_;
  Teuchos::RCP<map_type> ownedMap_;
  Teuchos::RCP<map_type> sharedMap_;
  Teuchos::RCP<map_type> ownedAndSharedMap_;
  Teuchos::RCP<import_type> importer_;
  Teuchos::RCP<export_type> exporter_;
  mutable Teuchos::RCP<mv_type> ownedAndSharedSln_;
  mutable Teuchos::RCP<mv_type> sharedRHS_;
  mutable Teuchos::RCP<mv_type> ownedAndSharedSln1D_;
  mutable Teuchos::RCP<mv_type> sharedRHS1D_;
};












template <typename InteriorOperatorType, typename BoundaryOperatorType, typename ExecSpace = Kokkos::DefaultHostExecutionSpace>
class MFOperator final : public operator_type
{
public:
  static constexpr int ndof = InteriorOperatorType::ndof;

  using scalar_type = double;
  using local_ordinal_type = int;
  using global_ordinal_type = long;
  using node_type = typename operator_type::node_type;

  MFOperator(InteriorOperatorType inmfInterior, BoundaryOperatorType inmfBoundary, Teuchos::RCP<map_type> map)
  : mfInterior(inmfInterior), mfBoundary(inmfBoundary), rowMap_(map)
  {}

  void apply(const mv_type& X, mv_type& Y, Teuchos::ETransp = Teuchos::NO_TRANS, double = 1.0, double = 0.0) const final  {
    Y.putScalar(0.0);
    const auto xv = X.getLocalView<ExecSpace>();
    auto yv = Y.getLocalView<ExecSpace>();
    mfInterior.compute_linearized_residual(xv, yv);
  }

  void compute_rhs(const mv_type& X, mv_type& Y) const
  {
    Y.putScalar(0.0);
    const auto xv = X.getLocalView<ExecSpace>();
    auto yv = Y.getLocalView<ExecSpace>();
    mfInterior.compute_rhs(yv);
  }

  Teuchos::RCP<const Tpetra::Map<int, long>> getDomainMap() const final { return rowMap_; }
  Teuchos::RCP<const Tpetra::Map<int, long>> getRangeMap() const final { return rowMap_; }
private:
  InteriorOperatorType mfInterior;
  BoundaryOperatorType mfBoundary;
  Teuchos::RCP<Tpetra::Map<int, long>> rowMap_;
};




} // namespace nalu
} // namespace Sierra

#endif

