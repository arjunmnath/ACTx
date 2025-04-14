#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

void test_div_by_zero() {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {1, 0, 1, 0};
  std::vector<int> shape = {2, 2};

  Tensor tensor1 = Tensor(data1, shape);
  Tensor tensor2 = Tensor(data2, shape);

  try {
    Tensor result = tensor1.div(&tensor2, false);
    assert(false && "Division by zero should throw an exception");
  } catch (const std::runtime_error &e) {
    assert(true && "Caught expected division by zero exception");
  }
}

void test_div_operation() {
  std::vector<float> data1 = {10, 20, 30, 40};
  std::vector<float> data2 = {2, 4, 6, 8};
  std::vector<int> shape = {2, 2};

  Tensor tensor1 = Tensor(data1, shape);
  Tensor tensor2 = Tensor(data2, shape);

  Tensor result = tensor1.div(&tensor2, false);
  std::vector<float> exp = {5, 5, 5, 5};

  Tensor expected = Tensor(exp, shape);
  assert(result.logical_e(&expected).all() && "Addition failed!");
}

int main() {
  test_div_operation();
  test_div_by_zero();
  return 0;
}
