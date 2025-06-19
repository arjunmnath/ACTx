
#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

// ---- ADD ----
TEST(TensorAutodiff, AddBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> y_data = {5.0, 6.0, 7.0, 8.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data, shape, DType::float32, true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data, shape, DType::float32, true, DeviceType::MPS);

  Tensor *z = x->add(y, false);
  z->backward();
  std::vector<float> vals = {1, 1, 1, 1};
  Tensor *expected = new Tensor(vals, shape);
  EXPECT_TRUE(x->grad->logical_e(expected)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(expected)->all()) << "y grad incorrect";
}

// ---- SUB ----
TEST(TensorAutodiff, SubBackwardWorks) {
  std::vector<float> x_data = {10.0, 20.0, 30.0, 40.0};
  std::vector<float> y_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data, shape, DType::float32, true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data, shape, DType::float32, true, DeviceType::MPS);

  Tensor *z = x->sub(y, false);
  z->backward();

  Tensor *ones = Tensor::ones(shape);
  Tensor *zeros = Tensor::zeros(shape);
  Tensor *neg_ones = zeros->sub(ones, false);

  EXPECT_TRUE(x->grad->logical_e(ones)->all()) << "x grad incorrect";
  EXPECT_TRUE(y->grad->logical_e(neg_ones)->all()) << "y grad incorrect";
}
/*
// ---- MUL ----
TEST(TensorAutodiff, MulBackwardWorks) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> y_data = {10.0, 20.0, 30.0, 40.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data, shape, DType::float32, true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data, shape, DType::float32, true, DeviceType::MPS);

  Tensor *z = x->mul(y, false);
  z->backward();

  Tensor *expected_x_grad = new Tensor(y_data, shape);
  Tensor *expected_y_grad = new Tensor(x_data, shape);

  EXPECT_TRUE(x->grad_tensor->logical_e(expected_x_grad)->all())
      << "x grad incorrect";
  EXPECT_TRUE(y->grad_tensor->logical_e(expected_y_grad)->all())
      << "y grad incorrect";
}

// ---- DIV ----
TEST(TensorAutodiff, DivBackwardWorks) {
  std::vector<float> x_data = {8.0, 18.0, 32.0, 50.0};
  std::vector<float> y_data = {2.0, 3.0, 4.0, 5.0};
  std::vector<int> shape = {2, 2};

  Tensor *x = new Tensor(x_data, shape, DType::float32, true, DeviceType::MPS);
  Tensor *y = new Tensor(y_data, shape, DType::float32, true, DeviceType::MPS);

  Tensor *z = x->div(y, false);
  z->backward();

  std::vector<float> dx_data = {1.0f / 2, 1.0f / 3, 1.0f / 4, 1.0f / 5};
  std::vector<float> dy_data = {-8.0f / (2 * 2), -18.0f / (3 * 3),
                                -32.0f / (4 * 4), -50.0f / (5 * 5)};
  Tensor *expected_dx = new Tensor(dx_data, shape);
  Tensor *expected_dy = new Tensor(dy_data, shape);

  EXPECT_TRUE(x->grad_tensor->logical_e(expected_dx)->all())
      << "x grad incorrect";
  EXPECT_TRUE(y->grad_tensor->logical_e(expected_dy)->all())
      << "y grad incorrect";
}
*/
