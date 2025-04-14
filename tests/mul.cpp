#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {2, 3, 4, 5};
  std::vector<int> shape = {2, 2};

  Tensor tensor1 = Tensor(data1, shape);
  Tensor tensor2 = Tensor(data2, shape);

  Tensor result = tensor1.mul(&tensor2, false);
  std::vector<float> exp = {2, 6, 12, 20};

  Tensor expected = Tensor(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;
}
