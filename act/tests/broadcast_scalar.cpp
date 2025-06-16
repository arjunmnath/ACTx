#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorBroadcastScalar, ScalarBroadcastingAddition) {
  std::vector<float> data1 = {10};
  std::vector<float> data2 = {1, 2, 3, 4};
  std::vector<int> shape1 = {1};
  std::vector<int> shape2 = {2, 2};

  Tensor *tensor1 = new Tensor(data1, shape1);
  Tensor *tensor2 = new Tensor(data2, shape2);

  Tensor *result = tensor1->add(tensor2, false);

  std::vector<float> expected_data = {11, 12, 13, 14};
  Tensor *expected = new Tensor(expected_data, shape2);

  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Scalar broadcasting failed";
}
