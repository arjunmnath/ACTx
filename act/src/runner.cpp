#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int> shape = {3, 4};
  std::vector<Slice> slices = {Slice(1, 3, 1), Slice(1, 4, 2)};
  Tensor *tensor = new Tensor(data, shape);

  // Slice [1:3, 1:4:2]
  Tensor *result = tensor->view(slices);

  // Expected:
  // row 1, col 1
  // row 1, col 3
  // row 2, col 1
  // row 2, col 3
  std::vector<float> expected_data = {5, 7, 9, 11};
  Tensor *expected = new Tensor(expected_data, {2, 2});
  tensor->print();
  result->print();
  expected->print();
  Tensor *logical_e = result->logical_e(expected);
  logical_e->print();
  std::cout << logical_e->all() << " ";
  return 0;
}
