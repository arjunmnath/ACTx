#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorBroadcastSingleAxes, SingleAxisBroadcastingAddition) {
  std::vector<float> data1 = {1, 2, 3};
  std::vector<float> data2 = {10};
  std::vector<int> shape1 = {3};
  std::vector<int> shape2 = {1};

  Tensor tensor1(data1, shape1);
  Tensor tensor2(data2, shape2);

  Tensor result = tensor1.add(&tensor2, false);
  std::vector<float> expected_data = {11, 12, 13};
  Tensor expected(expected_data, shape1);

  EXPECT_TRUE(result.logical_e(&expected).all())
      << "Single-axis broadcasting failed";
}
