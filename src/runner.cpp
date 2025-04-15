#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape1 = {3, 4};
  std::vector<int> shape2 = {1};

  Tensor tensor1 = Tensor::ones(shape1);
  Tensor tensor2 = Tensor::ones(shape2);
  tensor2.print_matrix();
  Tensor result = tensor1.add(&tensor2, false);
  result.print_matrix();
  return 0;
}
