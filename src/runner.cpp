#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape = {2, 3};
  Tensor a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0, 0, 0};
  Tensor expected(ones, shape, DType::int32);
  a.print();
  expected.print();
  Tensor b = a.logical_e(&expected);
  b.print();
  std::cout << b.all() << " " << std::endl;
  return 0;
}
