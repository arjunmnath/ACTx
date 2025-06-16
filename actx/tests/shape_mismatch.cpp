#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

/*
TEST(TensorTest, MatmulShapeMismatchThrows) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};
  std::vector<int> shape1 = {2, 2};
  std::vector<int> shape2 = {3, 1};

  Tensor tensor1(data1, shape1);
  Tensor tensor2(data2, shape2);

  EXPECT_THROW({
    tensor1.matmul(&tensor2);
  }, std::runtime_error);
}
*/

TEST(TensorShape, BroadcastAddOperationValid) {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6};
  std::vector<int> shape1 = {2, 2};
  std::vector<int> shape2 = {1, 2};

  Tensor *tensor1 = new Tensor(data1, shape1);
  Tensor *tensor2 = new Tensor(data2, shape2);

  EXPECT_NO_THROW({ tensor1->add(tensor2, false); });
}
