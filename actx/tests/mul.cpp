#include "tensor.h"
#include "utility.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorMultiplication, MultiplicationWorks) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {2, 3, 4, 5};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = make_tensor(data1, shape);
  Tensor *tensor2 = make_tensor(data2, shape);

  Tensor *result = tensor1->mul(tensor2, false);
  std::vector<float> expected_data = {2, 6, 12, 20};
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor multiplication failed";
}

TEST(TensorMultiplication, Negation) {
  std::vector<float> data2 = {2, 3, 4, 5};
  std::vector<int> shape = {2, 2};
  Tensor *tensor2 = make_tensor(data2, shape);
  Tensor *result = tensor2->negate(true);
  std::vector<float> expected_data = {-2, -3, -4, -5};
  Tensor *expected = make_tensor(expected_data, shape);
  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor Negation failed";
}
