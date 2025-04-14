#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data1 = {1, 2, 3};
  std::vector<float> data2 = {4, 5};
  std::vector<int> shape1 = {3};
  std::vector<int> shape2 = {2};

  Tensor tensor1 = Tensor(data1, shape1);
  Tensor tensor2 = Tensor(data2, shape2);

  try {
    Tensor result = tensor1.add(&tensor2, false);
    assert(false && "Incompatible shapes did not throw an error!");
  } catch (const std::invalid_argument &e) {
    assert(true && "Incompatible shapes correctly threw an error.");
  }
  return 0;
}
