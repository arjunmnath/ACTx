#include "main.h"
#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>
Tensor *make_tensor(std::vector<float> data, std::vector<int> shape = {2}) {
  Tensor *t = new Tensor(data, shape, DType::float32, false);
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
  //

  Tensor *a = make_tensor({10.0f, 20.0f});
  Tensor *b = make_tensor({2.0f, 5.0f});
  Tensor *result = make_tensor({0.0f, 0.0f});
  a->print();
  b->print();
  result->print();
  dispatcher->call(OPType::DIV, DeviceType::MPS, {a, b, result});

  // Tensor *a = Tensor::eye(5);
  // Tensor *b = Tensor::clone(a);
  // Tensor *c = Tensor::ones(std::vector<int>{5, 5});
  // Tensor *d = Tensor::full(std::vector<int>{5, 5}, -1.0f);
  // a->add(c, true);
  // a->sub(c, true);
  // a->print();
  // b->print();
  //
  // Memory *a = pool->request_memory(DeviceType::MPS, 512, DType::float32);
  // a->storage->metal.label = [NSString stringWithUTF8String:"a"];
  // Memory *b = pool->request_memory(DeviceType::MPS, 512, DType::float32);
  // Memory::copy(a, b);
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
