#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3};
  std::vector<float> data2 = {10};
  std::vector<int> shape1 = {3};
  std::vector<int> shape2 = {1};

  Tensor tensor1 = Tensor(data1, shape1);
  Tensor tensor2 = Tensor(data2, shape2);
  Tensor result = tensor1.add(&tensor2, false);

  std::vector<float> exp = {11, 12, 13};
  Tensor expected = Tensor(exp, shape1);
  assert(result.logical_e(&expected).all() &&
         "Single-axis broadcasting failed!");
  return 0;
}
