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
  // ——— Initialization ———
  // a: ones, b: full of 3s, c: eye(2)
  Tensor *a = Tensor::full({2, 2}, 4.2343f, DType::float32,
                           /*req_grad=*/true, DeviceType::MPS);
  Tensor *b = Tensor::full({2, 2}, 1.2344f, DType::float32,
                           /*req_grad=*/true, DeviceType::MPS);
  Tensor *c =
      Tensor::eye(2, DType::float32, /*req_grad=*/false, DeviceType::MPS);
  // ——— Build the graph ———
  Tensor *d1 = a->negate();                    // d1 = -a
  Tensor *d2 = d1->add(b);                     // d2 = b + (-a)
  Tensor *d3 = d2->sub(Tensor::zeros_like(a)); // d3 = d2 - 0
  Tensor *d4 = d3->mul(a);                     // d4 = d3 * a
  Tensor *d5 = d4->div(b);                     // d5 = d4 / b
  Tensor *d6 = d5->pow(2.0f);                  // d6 = (d5)^2
  // Tensor *d7 = d6->matmul(c);                  // d7 = d6 @ I
  Tensor *d7 = d6->div(c);    // d7 = d6 @ I
  Tensor *d8 = d7->clone(d7); // d8 = deep copy of d7

  // ——— Math & logs ———
  Tensor *m1 = d8->exp();
  Tensor *m2 = m1->sqrt();
  Tensor *m3 = m2->log();
  Tensor *m4 = m3->log2();
  Tensor *m5 = m4->log10();

  m5->backward();
  // a->grad->print();
  b->grad->print();
  // EXPECT_TRUE(a->grad) << "gradient not set";
  // EXPECT_TRUE(b->grad) << "gradient not set";

  // Clean up
  // delete a;
  // delete b;
  // delete c;
  // delete d1;
  // delete d2;
  // delete d3;
  // delete d4;
  // delete d5;
  // delete d6;
  // delete d7;
  // delete d8;
  // delete m1;
  // delete m2;
  // delete m3;
  // delete m4;
  // delete m5;
}
