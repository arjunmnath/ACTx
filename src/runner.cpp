#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape1 = {3, 4};
  std::vector<int> shape2 = {1};

  Tensor tensor1 = Tensor::ones(shape1, DType::float32);
  Tensor tensor2 = Tensor::ones(shape2, DType::float32);

  tensor1.print();
  tensor2.print();
  // FIX: operations going out of bound
  Tensor result = tensor1.add(&tensor2, false);
  result.print();
  return 0;
}
