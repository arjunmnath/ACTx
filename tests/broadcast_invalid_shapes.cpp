#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3};
  std::vector<float> data2 = {4, 5};
  std::vector<int> shape1 = {3};
  std::vector<int> shape2 = {2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);

  try {
    Tensor<float> result = tensor1.add(&tensor2, false);
    assert(false && "Incompatible shapes did not throw an error!");
  } catch (const std::invalid_argument &e) {
    assert(true && "Incompatible shapes correctly threw an error.");
  }
  return 0;
}
