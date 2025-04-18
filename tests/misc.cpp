
#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>

TEST(TensorTest, EmptyTensorThrows) {
  std::vector<float> data1 = {};
  std::vector<float> data2 = {};
  std::vector<int> shape = {};

  EXPECT_THROW({
    Tensor tensor1(data1, shape);
    Tensor tensor2(data2, shape);
  }, std::runtime_error);
}
