#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorPow, PowerOperationWorks) {
  std::vector<float> data = {2, 3, 4, 5};
  std::vector<float> expected_data = {4, 9, 16, 25}; // squared
  std::vector<int> shape = {2, 2};

  Tensor *tensor = new Tensor(data, shape);
  Tensor *result = tensor->pow(2.0f, false);
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor power operation failed";
}

TEST(TensorPow, PowerZeroGivesOne) {
  std::vector<float> data = {2, 3, 4, 5};
  std::vector<float> expected_data = {1, 1, 1, 1}; // anything ^ 0 = 1
  std::vector<int> shape = {2, 2};

  Tensor *tensor = new Tensor(data, shape);
  Tensor *result = tensor->pow(0.0f, false);
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor zero power failed";
}

TEST(TensorPow, CubeRootPowerWorks) {
  std::vector<float> data = {8, 27, 64, 125};
  std::vector<float> expected_data = {2, 3, 4, 5}; // cube roots
  std::vector<int> shape = {2, 2};

  Tensor *tensor = new Tensor(data, shape);
  Tensor *result = tensor->pow(1.0f / 3.0f, false);
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor cube root power failed";
}
