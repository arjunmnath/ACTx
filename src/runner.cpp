#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape1 = {3, 4};
  std::vector<int> shape2 = {1};

  Tensor<float> tensor1 = Tensor<float>::ones(shape1);
  Tensor<float> tensor2 = Tensor<float>::ones(shape2);
  tensor2.print_matrix();
  Tensor<float> result = tensor1.add(&tensor2, false);
  result.print_matrix();
  return 0;
}
