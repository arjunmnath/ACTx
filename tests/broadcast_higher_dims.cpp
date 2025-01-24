#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2};
  std::vector<float> data2 = {10, 20, 30, 40};
  std::vector<int> shape1 = {2, 1};
  std::vector<int> shape2 = {2, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);
  Tensor<float> result = tensor1.add(&tensor2, false);

  std::vector<float> exp = {11, 21, 32, 42};
  Tensor<float> expected = Tensor<float>(exp, shape2);
  assert(result.logical_e(&expected).all() && "Higher-dimensional broadcasting failed!");
  return 0;
}
