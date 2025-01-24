#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {10};
  std::vector<float> data2 = {1, 2, 3, 4};
  std::vector<int> shape1 = {};
  std::vector<int> shape2 = {2, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);
  Tensor<float> result = tensor1.add(&tensor2, false);

  std::vector<float> exp = {11, 12, 13, 14};
  Tensor<float> expected = Tensor<float>(exp, shape2);
  assert(result.logical_e(&expected).all() && "Scalar broadcasting failed!");
  return 0;
}
