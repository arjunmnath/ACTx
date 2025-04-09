#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape1 = {3, 1};
  std::vector<int> shape2 = {1, 4};

  Tensor<float> tensor1 = Tensor<float>::ones(shape1);
  Tensor<float> tensor2 = Tensor<float>::ones(shape2);
  Tensor<float> result = tensor1.add(&tensor2, true);
  result.print_matrix();
  /*std::vector<float> exp = {4, 6, 5, 7};*/
  /*Tensor<float> expected = Tensor<float>(exp, shape2);*/
  /*assert(result.logical_e(&expected).all() && "Broadcasting addition
   * failed!");*/
  return 0;
}
