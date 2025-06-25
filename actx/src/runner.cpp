#include "main.h"
#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>
Tensor *make_tensor(std::vector<float> data, std::vector<int> shape = {2}) {
  Tensor *t =
      new Tensor(data.data(), data.size(), shape, DType::float32, false);
  return t;
}
int main() {
  // std::vector<float> vals = {1.0f};
  // std::vector<int> dims = {1};
  // Tensor *t1 = new Tensor(vals, dims);
  // t1->requires_grad = true;
  // Tensor *t2 = new Tensor(vals, dims);
  // Tensor *t3 = t1->add(t2, false); // t3 = t1 + t2
  // std::vector<float> vals2 = {4.0f};
  // Tensor *t4 = new Tensor(vals2, dims);
  // Tensor *t5 = t3->mul(t4, false); // t5 = t3 * t4
  // std::vector<float> vals3 = {6.0f};
  // Tensor *t6 = new Tensor(vals3, dims);
  // Tensor *t7 = t5->sub(t6, false); // t7 = t5 - t6
  // std::vector<float> vals4 = {8.0f};
  // Tensor *t8 = new Tensor(vals4, dims);
  // Tensor *t9 = t7->div(t8, false); // t9 = t7 / t8
  // t9->backward();
}
