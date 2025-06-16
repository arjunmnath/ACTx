
#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorSubtraction, SubtractionWorks) {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {1, 2, 3, 4};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = new Tensor(data1, shape);
  Tensor *tensor2 = new Tensor(data2, shape);

  Tensor *result = tensor1->sub(tensor2, false);
  std::vector<float> expected_data = {9, 18, 27, 36};
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor subtraction failed";
}
