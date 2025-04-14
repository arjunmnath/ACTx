#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {1, 2, 3, 4};
  std::vector<int> shape = {2, 2};

  Tensor tensor1 = Tensor(data1, shape);
  Tensor tensor2 = Tensor(data2, shape);

  Tensor result = tensor1.sub(&tensor2, false);
  std::vector<float> exp = {9, 18, 27, 36};

  Tensor expected = Tensor(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;

}
