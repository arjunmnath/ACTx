#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

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
  //

  Tensor *a = Tensor::eye(5);
  Tensor *b = Tensor::clone(a);
  Tensor *c = Tensor::ones(std::vector<int>{5, 5});
  Tensor *d = Tensor::full(std::vector<int>{5, 5}, -1.0f);
  a->memory->storage->metal.label = [NSString stringWithUTF8String:"a(eye)"];
  a->add(c, true);
  a->sub(c, true);
  b->print();

  // std::vector<float> x_data = {10.0, 20.0, 30.0, 40.0};
  // std::vector<float> y_data = {1.0, 2.0, 3.0, 4.0};
  // std::vector<int> shape = {2, 2};
  //
  // Tensor *x = new Tensor(x_data, shape, DType::float32, true,
  // DeviceType::MPS); Tensor *y = new Tensor(y_data, shape, DType::float32,
  // true, DeviceType::MPS);
  //
  // Tensor *z = x->sub(y, false);
  // z->backward();
  // Tensor *neg_ones = Tensor::full(shape, -1.0f);
  // x->grad->print();
  // y->grad->print();
  //
  // EXPECT_TRUE(x->grad->logical_e(ones)->all()) << "x grad incorrect";
  // EXPECT_TRUE(y->grad->logical_e(neg_ones)->all()) << "y grad incorrect";
  //
  return 0;
}
