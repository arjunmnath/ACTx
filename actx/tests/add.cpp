#include "tensor.h"
#include "utility.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorAddition, AdditionWorks) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = make_tensor(data1, shape);
  Tensor *tensor2 = make_tensor(data2, shape);
  Tensor *result = tensor1->add(tensor2, false);
  std::vector<float> expected_data = {6, 8, 10, 12};
  Tensor *expected = make_tensor(expected_data, shape);
  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor addition failed";
}
