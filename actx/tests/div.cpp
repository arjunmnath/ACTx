#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorDivion, DivisionOperationWorks) {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {2, 4, 6, 8};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = new Tensor(data1, shape);
  Tensor *tensor2 = new Tensor(data2, shape);

  Tensor *result = tensor1->div(tensor2, false);
  std::vector<float> expected_data = {5, 5, 5, 5};
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor division failed";
}

TEST(TensorDivion, DivisionByZeroThrows) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {1, 0, 1, 0};
  std::vector<int> shape = {2, 2};

  Tensor *tensor1 = new Tensor(data1, shape);
  Tensor *tensor2 = new Tensor(data2, shape);

  EXPECT_THROW({ tensor1->div(tensor2, false); }, std::runtime_error);
}
