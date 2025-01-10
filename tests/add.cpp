#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};
  std::vector<int> shape = {2, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape);
  Tensor<float> tensor2 = Tensor<float>(data2, shape);
  Tensor<float> result = tensor1.add(&tensor2, false);

  std::vector<float> exp = {6, 8, 10, 12};
  Tensor<float> expected = Tensor<float>(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;
}
