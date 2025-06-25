#include "tensor.h"
#include "utility.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorDivion, DivisionOperation) {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {2, 4, 6, 8};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = make_tensor(data1, shape);
  Tensor *tensor2 = make_tensor(data2, shape);

  Tensor *result = tensor1->div(tensor2, false);
  std::vector<float> expected_data = {5, 5, 5, 5};
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor division failed";
}

TEST(TensorDivion, DivisionByZero) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {1, 0, 1, 0};
  std::vector<int> shape = {2, 2};
  Tensor *tensor1 = make_tensor(data1, shape);
  Tensor *tensor2 = make_tensor(data2, shape);
  Tensor *result = tensor1->div(tensor2, false);
  EXPECT_TRUE(std::isinf(result->_get_element(1)));
  EXPECT_TRUE(std::isinf(result->_get_element(3)));
}
