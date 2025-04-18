#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorBroadcastShapes, BroadcastingAdditionWithOnes) {
  std::vector<int> shape1 = {3, 1};
  std::vector<int> shape2 = {1, 4};

  Tensor tensor1 = Tensor::ones(shape1);
  Tensor tensor2 = Tensor::ones(shape2);

  Tensor result = tensor1.add(&tensor2, false);

  std::vector<float> expected_data = {4, 6, 5, 7};
  Tensor expected(expected_data, shape2);

  EXPECT_TRUE(result.logical_e(&expected).all())
      << "Broadcasting addition failed";
}
