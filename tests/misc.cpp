#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

void test_empty_tensor() {
  std::vector<float> data1 = {};
  std::vector<float> data2 = {};
  std::vector<int> shape = {};

  Tensor<float> tensor1 = Tensor<float>(data1, shape);
  Tensor<float> tensor2 = Tensor<float>(data2, shape);

  Tensor<float> result = tensor1.add(&tensor2, false);

  assert(result.size == 0 && "Result of adding empty tensors should be empty");
}

int main() {
  test_empty_tensor();
  return 0;
}
