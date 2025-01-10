#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3, 4, 5, 6};
  std::vector<float> data2 = {7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3};
  std::vector<int> shape2 = {3, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);

  Tensor<float> result = tensor1.matmul(&tensor2);
  std::vector<float> exp = {58, 64, 139, 154};

  std::vector<int> shape = {2, 2};
  Tensor<float> expected = Tensor<float>(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
  return 0;
}
