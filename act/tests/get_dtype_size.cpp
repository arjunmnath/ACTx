
#include "types.h"
#include "utility.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(TensorDataType, Int8Size) { EXPECT_EQ(getDTypeSize(DType::int8), 1); }

TEST(TensorDataType, Float16Size) {
  EXPECT_EQ(getDTypeSize(DType::float16), 2);
}

TEST(TensorDataType, Int16Size) { EXPECT_EQ(getDTypeSize(DType::int16), 2); }

TEST(TensorDataType, Float32Size) {
  EXPECT_EQ(getDTypeSize(DType::float32), 4);
}

TEST(TensorDataType, Int32Size) { EXPECT_EQ(getDTypeSize(DType::int32), 4); }

TEST(TensorDataType, Int64Size) { EXPECT_EQ(getDTypeSize(DType::int64), 8); }

TEST(TensorDataType, InvalidTypeThrows) {
  EXPECT_THROW(getDTypeSize(static_cast<DType>(999)), std::invalid_argument);
}
