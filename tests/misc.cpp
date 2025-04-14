#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

void test_empty_tensor() {
  std::vector<float> data1 = {};
  std::vector<float> data2 = {};
  std::vector<int> shape = {};

  try {
    Tensor tensor1 = Tensor(data1, shape);
    Tensor tensor2 = Tensor(data2, shape);

    assert(false && "empty tensor created was allowed");
  } catch (const std::runtime_error &e) {

    assert(true && "ok");
  }
}

int main() {
  test_empty_tensor();
  return 0;
}
