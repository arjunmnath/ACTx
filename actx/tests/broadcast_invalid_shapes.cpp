#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorBroadcastInvalidShape, IncompatibleShapesAddition) {
  std::vector<float> data1 = {1, 2, 3};
  std::vector<float> data2 = {4, 5};
  std::vector<int> shape1 = {3};
  std::vector<int> shape2 = {2};

  Tensor *tensor1 = new Tensor(data1, shape1);
  Tensor *tensor2 = new Tensor(data2, shape2);

  try {
    Tensor *result = tensor1->add(tensor2, false);
    FAIL() << "Incompatible shapes did not throw an error!";
  } catch (const std::invalid_argument &e) {
    SUCCEED() << "Incompatible shapes correctly threw an error.";
  }
}
