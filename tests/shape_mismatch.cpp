#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <cassert>
#include <vector>

void test_matmul_shape_mismatch() {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6, 7, 8};
  std::vector<int> shape1 = {2, 2};
  std::vector<int> shape2 = {3, 1};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);

  try {
    Tensor<float> result = tensor1.matmul(&tensor2);
    assert(false && "Matrix multiplication with mismatched shapes should throw "
                    "an exception");
  } catch (const std::runtime_error &e) {
    assert(true && "Caught expected shape mismatch exception for matmul");
  }
}

void test_matmul_operations() {
  std::vector<float> data1 = {1, 2, 3, 4};
  std::vector<float> data2 = {5, 6};
  std::vector<int> shape1 = {2, 2};
  std::vector<int> shape2 = {1, 2};

  Tensor<float> tensor1 = Tensor<float>(data1, shape1);
  Tensor<float> tensor2 = Tensor<float>(data2, shape2);

  try {
    Tensor<float> result = tensor1.add(&tensor2, false);
    assert(false && "Shape mismatch should throw an exception");
  } catch (const std::runtime_error &e) {
    assert(true && "Caught expected shape mismatch exception");
  }
}

int main() {
  test_matmul_operations();
  test_matmul_shape_mismatch();
  return 0;
}
