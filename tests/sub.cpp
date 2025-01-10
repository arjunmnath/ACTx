#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {1, 2, 3, 4};
  std::vector<int> shape = {2, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape);
  Tensor<float> tensor2 = Tensor<float>(data2, shape);

  Tensor<float> result = tensor1.sub(&tensor2, false);
  std::vector<float> exp = {9, 18, 27, 36};

  Tensor<float> expected = Tensor<float>(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;
}
