
#include "tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_PI_6 (M_PI / 6.0)
#define M_PI_3 (M_PI / 3.0)

// ---- ADD ----
TEST(TensorAutodiff, AddBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> y_data = {5.0, 6.0, 7.0, 8.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);

  Tensor *z = x->add(y, false);
  z->backward();
  std::vector<float> vals = {1, 1, 1, 1};
  Tensor *expected = new Tensor(vals.data(), vals.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(expected)->all()) << "y grad incorrect";
}

// ---- SUB ----
TEST(TensorAutodiff, SubBackwardWorks) {
  std::vector<float> x_data = {10.0, 20.0, 30.0, 40.0};
  std::vector<float> y_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int> shape = {2, 2};
  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->sub(y, false);
  z->backward();

  Tensor *ones = Tensor::ones(shape);
  Tensor *zeros = Tensor::zeros(shape);
  Tensor *neg_ones = zeros->sub(ones, false);

  EXPECT_TRUE(x->grad->logical_e(ones)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(neg_ones)->all()) << "y grad incorrect";
}
// ---- MUL ----
TEST(TensorAutodiff, MulBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> y_data = {10.0, 20.0, 30.0, 40.0};
  std::vector<int> shape = {2, 2};
  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->mul(y, false);
  z->backward();

  Tensor *expected_x_grad = new Tensor(y_data.data(), y_data.size(), shape);
  Tensor *expected_y_grad = new Tensor(x_data.data(), x_data.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_x_grad)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(expected_y_grad)->all()) << "y grad incorrect";
}
// ---- DIV ----
TEST(TensorAutodiff, DivBackwardWorks) {
  std::vector<float> x_data = {8.0, 18.0, 32.0, 50.0};
  std::vector<float> y_data = {2.0, 3.0, 4.0, 5.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->div(y, false);
  z->backward();

  std::vector<float> dx_data = {1.0f / 2, 1.0f / 3, 1.0f / 4, 1.0f / 5};
  std::vector<float> dy_data = {-8.0f / (2 * 2), -18.0f / (3 * 3),
                                -32.0f / (4 * 4), -50.0f / (5 * 5)};
  Tensor *expected_dx = new Tensor(dx_data.data(), dx_data.size(), shape);
  Tensor *expected_dy = new Tensor(dy_data.data(), dy_data.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(expected_dy)->all()) << "y grad incorrect";
}

TEST(TensorAutodiff, PowBackwardWorks) {
  std::vector<float> x_data = {2.0, 3.0, 4.0, 5.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->pow(2.0f); // z = x^2
  z->backward();

  std::vector<float> expected_grad = {2.0f * 2.0f, 2.0f * 3.0f, 2.0f * 4.0f,
                                      2.0f * 5.0f}; // dz/dx = 2x
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "pow grad incorrect";
}
TEST(TensorAutodiff, SqrtBackwardWorks) {
  std::vector<float> x_data = {4.0, 9.0, 16.0, 25.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->sqrt();
  z->backward();

  std::vector<float> expected_grad = {0.5f / 2.0f, 0.5f / 3.0f, 0.5f / 4.0f,
                                      0.5f / 5.0f};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "sqrt grad incorrect ";
}

TEST(TensorAutodiff, ExpBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->exp();
  z->backward();

  std::vector<float> expected_grad = {std::exp(1.0f), std::exp(2.0f),
                                      std::exp(3.0f), std::exp(4.0f)};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "exp grad incorrect";
}

TEST(TensorAutodiff, LogBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 4.0, 5.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->log();
  z->backward();

  std::vector<float> expected_grad = {1.0f / 1.0f, 1.0f / 2.0f, 1.0f / 4.0f,
                                      1.0f / 5.0f};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "log grad incorrect";
}
TEST(TensorAutodiff, Log2BackwardWorks) {
  std::vector<float> x_data = {2.0, 4.0, 8.0, 16.0};
  std::vector<int> shape = {2, 2};
  float ln2 = std::log(2.0f);

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->log2();
  z->backward();

  std::vector<float> expected_grad = {1.0f / (2.0f * ln2), 1.0f / (4.0f * ln2),
                                      1.0f / (8.0f * ln2),
                                      1.0f / (16.0f * ln2)};
  Tensor *expected_dx = new Tensor(expected_grad.data(), x_data.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "log2 grad incorrect";
}

TEST(TensorAutodiff, Log10BackwardWorks) {
  std::vector<float> x_data = {10.0, 100.0, 1000.0, 10000.0};
  std::vector<int> shape = {2, 2};
  float ln10 = std::log(10.0f);

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->log10();
  z->backward();

  std::vector<float> expected_grad = {
      1.0f / (10.0f * ln10), 1.0f / (100.0f * ln10), 1.0f / (1000.0f * ln10),
      1.0f / (10000.0f * ln10)};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "log10 grad incorrect";
}

TEST(TensorAutodiff, SinBackwardWorks) {
  std::vector<float> x_data = {0.0, M_PI_4, M_PI_2, M_PI};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->sin();
  z->backward();

  std::vector<float> expected_grad = {
      static_cast<float>(std::cos(0.0f)), static_cast<float>(std::cos(M_PI_4)),
      static_cast<float>(std::cos(M_PI_2)), static_cast<float>(std::cos(M_PI))};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "sin grad incorrect";
}

TEST(TensorAutodiff, CosBackwardWorks) {
  std::vector<float> x_data = {0.0, M_PI_4, M_PI_2, M_PI};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->cos();
  z->backward();

  std::vector<float> expected_grad = {-static_cast<float>(std::sin(0.0f)),
                                      -static_cast<float>(std::sin(M_PI_4)),
                                      -static_cast<float>(std::sin(M_PI_2)),
                                      -static_cast<float>(std::sin(M_PI))};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "cos grad incorrect";
}

TEST(TensorAutodiff, TanBackwardWorks) {
  std::vector<float> x_data = {0.0, M_PI_6, M_PI_4, M_PI_3};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->tan();
  z->backward();

  std::vector<float> expected_grad = {
      1.0f + static_cast<float>(std::pow(std::tan(0.0f), 2)),
      1.0f + static_cast<float>(std::pow(std::tan(M_PI_6), 2)),
      1.0f + static_cast<float>(std::pow(std::tan(M_PI_4), 2)),
      1.0f + static_cast<float>(std::pow(std::tan(M_PI_3), 2))};
  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);

  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "tan grad incorrect";
}

TEST(TensorAutodiff, AsinBackwardWorks) {
  std::vector<float> x_data = {0.0f, 0.5f, 0.7f, 0.9f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->asin();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(1.0f / std::sqrt(1.0f - v * v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "asin grad incorrect";
}

TEST(TensorAutodiff, AcosBackwardWorks) {
  std::vector<float> x_data = {0.0f, 0.5f, 0.7f, 0.9f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->acos();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(-1.0f / std::sqrt(1.0f - v * v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "acos grad incorrect";
}

TEST(TensorAutodiff, AtanBackwardWorks) {
  std::vector<float> x_data = {0.0f, 1.0f, 2.0f, 3.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->atan();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(1.0f / (1.0f + v * v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "atan grad incorrect";
}

TEST(TensorAutodiff, Atan2BackwardWorks) {
  std::vector<float> y_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> x_data = {2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int> shape = {2, 2};

  Tensor *y = new Tensor(y_data.data(), y_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->atan2(y);
  z->backward();

  std::vector<float> expected_dy, expected_dx;
  for (int i = 0; i < y_data.size(); ++i) {
    float dy = x_data[i] / (y_data[i] * y_data[i] + x_data[i] * x_data[i]);
    float dx = -y_data[i] / (y_data[i] * y_data[i] + x_data[i] * x_data[i]);
    expected_dy.push_back(dy);
    expected_dx.push_back(dx);
  }

  Tensor *dy = new Tensor(expected_dy.data(), expected_dy.size(), shape);
  Tensor *dx = new Tensor(expected_dx.data(), expected_dy.size(), shape);

  EXPECT_TRUE(y->grad->logical_e(dy)->all()) << "atan2 dy incorrect";
  EXPECT_TRUE(x->grad->logical_e(dx)->all()) << "atan2 dx incorrect";
}

TEST(TensorAutodiff, SinhBackwardWorks) {
  std::vector<float> x_data = {0.0f, 1.0f, -1.0f, 2.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->sinh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(std::cosh(v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "sinh grad incorrect";
}

TEST(TensorAutodiff, CoshBackwardWorks) {
  std::vector<float> x_data = {0.0f, 1.0f, -1.0f, 2.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->cosh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(std::sinh(v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "cosh grad incorrect";
}

TEST(TensorAutodiff, TanhBackwardWorks) {
  std::vector<float> x_data = {0.0f, 0.5f, -0.5f, 1.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->tanh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data) {
    float t = std::tanh(v);
    expected_grad.push_back(1.0f - t * t);
  }

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "tanh grad incorrect";
}

TEST(TensorAutodiff, AsinhBackwardWorks) {
  std::vector<float> x_data = {0.0f, 1.0f, 2.0f, -2.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->asinh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(1.0f / std::sqrt(v * v + 1.0f));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "asinh grad incorrect";
}

TEST(TensorAutodiff, AcoshBackwardWorks) {
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 5.0f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->acosh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(1.0f / std::sqrt(v * v - 1.0f));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "acosh grad incorrect";
}

TEST(TensorAutodiff, AtanhBackwardWorks) {
  std::vector<float> x_data = {0.0f, 0.2f, -0.5f, 0.7f};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data.data(), x_data.size(), shape, DType::float32,
                         true, DeviceType::MPS);
  Tensor *z = x->atanh();
  z->backward();

  std::vector<float> expected_grad;
  for (float v : x_data)
    expected_grad.push_back(1.0f / (1.0f - v * v));

  Tensor *expected_dx =
      new Tensor(expected_grad.data(), expected_grad.size(), shape);
  EXPECT_TRUE(x->grad->logical_e(expected_dx)->all()) << "atanh grad incorrect";
}

TEST(TensorAutodiff, ComparisonsThrowIfInGraph) {
  auto A = Tensor::ones({2, 2}, DType::float32, true);
  auto B = Tensor::zeros({2, 2}, DType::float32, true);

  auto c = A->logical_e(B);
  auto d = A->logical_ne(B);
  auto e = A->logical_gt(B);
  auto f = A->logical_gte(B);
  auto g = A->logical_lt(B);
  auto h = A->logical_lte(B);

  // c->requires_grad = d->requires_grad = e->requires_grad = f->requires_grad =
  // g->requires_grad = h->requires_grad = false;
  EXPECT_THROW({ c->backward(); }, std::logic_error);
  EXPECT_THROW({ d->backward(); }, std::logic_error);
  EXPECT_THROW({ e->backward(); }, std::logic_error);
  EXPECT_THROW({ f->backward(); }, std::logic_error);
  EXPECT_THROW({ g->backward(); }, std::logic_error);
  EXPECT_THROW({ h->backward(); }, std::logic_error);
}

TEST(TensorAutodiff, FullComputeGraphWorks) {
  // ——— Initialization ———
  // a: ones, b: full of 3s, c: eye(2)
  Tensor *a = Tensor::full({2, 2}, 4.2343f, DType::float32,
                           /*req_grad=*/true, DeviceType::MPS);
  Tensor *b = Tensor::full({2, 2}, 1.2344f, DType::float32,
                           /*req_grad=*/true, DeviceType::MPS);
  Tensor *c =
      Tensor::eye(2, DType::float32, /*req_grad=*/false, DeviceType::MPS);

  Tensor *epsilon = Tensor::full_like(b, 1e-4);
  epsilon->requires_grad = false;

  // ——— Build the graph ———
  Tensor *d1 = a->negate();
  Tensor *d2 = d1->add(b);
  Tensor *d3 = d2->sub(Tensor::zeros_like(a));
  Tensor *d4 = d3->mul(a);
  Tensor *d5 = d4->div(b->add(epsilon, true));
  Tensor *d6 = d5->pow(2.0f);
  // TODO: enable this matmul
  // Tensor *d7 = d6->matmul(c);
  Tensor *d7 = d6->div(c->add(epsilon, true));
  Tensor *d8 = Tensor::clone(d7);
  Tensor *m1 = d8->sqrt();
  Tensor *m2 = m1->log10();
  Tensor *m3 = m2->log();
  Tensor *m4 = m3->log2();
  Tensor *m5 = m4->exp();

  m5->backward();
  a->grad->print();
  b->grad->print();
  EXPECT_TRUE(a->grad) << "gradient not set";
  EXPECT_TRUE(b->grad) << "gradient not set";
  // Clean up
  delete a;
  delete b;
  delete c;
  delete epsilon;
  delete d1;
  delete d2;
  delete d3;
  delete d4;
  delete d5;
  delete d6;
  delete d7;
  delete d8;
  delete m1;
  delete m2;
  delete m3;
  delete m4;
  delete m5;
}
