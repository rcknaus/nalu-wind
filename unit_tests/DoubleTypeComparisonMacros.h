#ifndef _DoubleTypeComparisonMacros_h_
#define _DoubleTypeComparisonMacros_h_

#include <gtest/gtest.h>
#include "SimdInterface.h"

// this allows calling the macros with a double instead of a DoubleType
inline double get_simd_data_promote_double_to_doubletype(DoubleType val, int ln) { return stk::simd::get_data(val, ln); }

#define EXPECT_DOUBLETYPE_NEAR(val1, val2, abs_error)                                              \
  for (int ln = 0; ln < simdLen; ++ln)                                                    \
  EXPECT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat,                                   \
      get_simd_data_promote_double_to_doubletype(val1, ln),                                        \
      get_simd_data_promote_double_to_doubletype(val2, ln),                                        \
      abs_error)

#define EXPECT_DOUBLETYPE_EQ(val1, val2)                                                           \
  for (int ln = 0; ln < simdLen; ++ln)                                                    \
  EXPECT_DOUBLE_EQ(get_simd_data_promote_double_to_doubletype(val1, ln),                           \
      get_simd_data_promote_double_to_doubletype(val2, ln))

#define EXPECT_DOUBLETYPE_GTEQ(val1, val2)                                                         \
  for (int ln = 0; ln < simdLen; ++ln)                                                    \
  EXPECT_PRED_FORMAT2(::testing::DoubleLE,                                                         \
      get_simd_data_promote_double_to_doubletype(val2, ln),                                        \
      get_simd_data_promote_double_to_doubletype(val1, ln))

#define EXPECT_DOUBLETYPE_LTEQ(val1, val2)                                                         \
  for (int ln = 0; ln < simdLen; ++ln)                                                    \
  EXPECT_PRED_FORMAT2(::testing::DoubleLE,                                                         \
      get_simd_data_promote_double_to_doubletype(val1, ln),                                        \
      get_simd_data_promote_double_to_doubletype(val2, ln))



#endif

