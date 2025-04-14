#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape1 = {3, 1};
  std::vector<int> shape2 = {1, 4};

  Tensor tensor1 = Tensor::ones(shape1);
  Tensor tensor2 = Tensor::ones(shape2);
  Tensor result = tensor1.add(&tensor2, false);
  result.print_matrix();
  std::vector<float> exp = {4, 6, 5, 7};
  Tensor expected = Tensor(exp, shape2);
  assert(result.logical_e(&expected).all() && "Broadcasting addition failed!");
  return 0;
}
