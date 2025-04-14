#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

#include <iostream>
using namespace std;

int main() {
  std::vector<float> data1 = {1, 2, 3, 4, 5, 6};
  std::vector<float> data2 = {7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3};
  std::vector<int> shape2 = {3, 2};
  Tensor tensor1 = Tensor(data1, shape1);
  Tensor tensor2 = Tensor(data2, shape2);
  Tensor result = tensor1.matmul(&tensor2);
  std::vector<float> exp = {58, 64, 139, 154};
  std::vector<int> shape = {shape1[0], shape2[1]};
  Tensor expected = Tensor(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;
}
