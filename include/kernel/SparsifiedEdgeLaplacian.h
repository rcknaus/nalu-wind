#ifndef SparsifiedEdgeLaplacian_h
#define SparsifiedEdgeLaplacian_h

#include "LocalArray.h"
#include "KokkosInterface.h"
#include "SimdInterface.h"
#include "CVFEMTypeDefs.h"

namespace sierra { namespace nalu {

template <int p> LocalArray<DoubleType[p][p][p][8][8]>
sparsified_laplacian_lhs(const nodal_vector_view<p,DoubleType>&);

template <int p> LocalArray<DoubleType[p][p][p][12][2][2]>
sparsified_laplacian_edge_lhs(const nodal_vector_view<p,DoubleType>&);

LocalArray<double[8][8]> laplacian_lhs(const double box[3][8]);

} // namespace nalu
} // namespace Sierra

#endif

