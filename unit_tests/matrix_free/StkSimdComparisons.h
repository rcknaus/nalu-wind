#ifndef STK_SIMD_COMPARISONS_H
#define STK_SIMD_COMPARISONS_H

#include "gtest/gtest.h"

#include "matrix_free/KokkosFramework.h"

// this allows calling the macros with a double instead of a DoubleType
inline double
get_simd_data_promote_double_to_doubletype(
  sierra::nalu::matrix_free::ftype val, int ln)
{
  return stk::simd::get_data(val, ln);
}

#define EXPECT_DOUBLETYPE_NEAR(val1, val2, abs_error)                          \
  for (int ln = 0; ln < simd_len; ++ln)                                        \
  EXPECT_PRED_FORMAT3(                                                         \
    ::testing::internal::DoubleNearPredFormat,                                 \
    get_simd_data_promote_double_to_doubletype(val1, ln),                      \
    get_simd_data_promote_double_to_doubletype(val2, ln), abs_error)

#define ASSERT_DOUBLETYPE_NEAR(val1, val2, abs_error)                          \
  for (int ln = 0; ln < simd_len; ++ln)                                        \
  ASSERT_PRED_FORMAT3(                                                         \
    ::testing::internal::DoubleNearPredFormat,                                 \
    get_simd_data_promote_double_to_doubletype(val1, ln),                      \
    get_simd_data_promote_double_to_doubletype(val2, ln), abs_error)

#define EXPECT_DOUBLETYPE_EQ(val1, val2)                                       \
  for (int ln = 0; ln < simd_len; ++ln)                                        \
  EXPECT_DOUBLE_EQ(                                                            \
    get_simd_data_promote_double_to_doubletype(val1, ln),                      \
    get_simd_data_promote_double_to_doubletype(val2, ln))

#define EXPECT_DOUBLETYPE_GTEQ(val1, val2)                                     \
  for (int ln = 0; ln < simd_len; ++ln)                                        \
  EXPECT_PRED_FORMAT2(                                                         \
    ::testing::DoubleLE, get_simd_data_promote_double_to_doubletype(val2, ln), \
    get_simd_data_promote_double_to_doubletype(val1, ln))

#define EXPECT_DOUBLETYPE_LTEQ(val1, val2)                                     \
  for (int ln = 0; ln < simd_len; ++ln)                                        \
  EXPECT_PRED_FORMAT2(                                                         \
    ::testing::DoubleLE, get_simd_data_promote_double_to_doubletype(val1, ln), \
    get_simd_data_promote_double_to_doubletype(val2, ln))

#endif
