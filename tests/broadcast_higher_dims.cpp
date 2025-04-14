#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2};
  std::vector<float> data2 = {10, 20, 30, 40};
  std::vector<int> shape1 = {2, 1};
  std::vector<int> shape2 = {2, 2};

  Tensor tensor1 = Tensor(data1, shape1);
  Tensor tensor2 = Tensor(data2, shape2);
  Tensor result = tensor1.add(&tensor2, false);

  std::vector<float> exp = {11, 21, 32, 42};
  Tensor expected = Tensor(exp, shape2);
  assert(result.logical_e(&expected).all() &&
         "Higher-dimensional broadcasting failed!");
  return 0;
}
