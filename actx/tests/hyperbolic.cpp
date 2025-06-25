#include "tensor.h"
#include "utility.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorTrig, SinhOperation) {
  std::vector<float> data = {0.0, 1.0, -1.0, 2.0};
  std::vector<float> expected_data = {
      static_cast<float>(std::sinh(0.0)), static_cast<float>(std::sinh(1.0)),
      static_cast<float>(std::sinh(-1.0)), static_cast<float>(std::sinh(2.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->sinh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "sinh() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, CoshOperation) {
  std::vector<float> data = {0.0, 1.0, -1.0, 2.0};
  std::vector<float> expected_data = {
      static_cast<float>(std::cosh(0.0)), static_cast<float>(std::cosh(1.0)),
      static_cast<float>(std::cosh(-1.0)), static_cast<float>(std::cosh(2.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->cosh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "cosh() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, TanhOperation) {
  std::vector<float> data = {0.0, 1.0, -1.0, 2.0};
  std::vector<float> expected_data = {
      static_cast<float>(std::tanh(0.0)), static_cast<float>(std::tanh(1.0)),
      static_cast<float>(std::tanh(-1.0)), static_cast<float>(std::tanh(2.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->tanh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "tanh() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AsinhOperation) {
  std::vector<float> data = {0.0, 1.0, -1.0, 2.0};
  std::vector<float> expected_data = {static_cast<float>(std::asinh(0.0)),
                                      static_cast<float>(std::asinh(1.0)),
                                      static_cast<float>(std::asinh(-1.0)),
                                      static_cast<float>(std::asinh(2.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->asinh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "asinh() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AcoshOperation) {
  std::vector<float> data = {1.0, 2.0, 3.0, 10.0};
  std::vector<float> expected_data = {static_cast<float>(std::acosh(1.0)),
                                      static_cast<float>(std::acosh(2.0)),
                                      static_cast<float>(std::acosh(3.0)),
                                      static_cast<float>(std::acosh(10.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->acosh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "acosh() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AtanhOperation) {
  std::vector<float> data = {0.0f, 0.5f, -0.5f, 0.9f};
  std::vector<float> expected_data = {static_cast<float>(std::atanh(0.0f)),
                                      static_cast<float>(std::atanh(0.5f)),
                                      static_cast<float>(std::atanh(-0.5f)),
                                      static_cast<float>(std::atanh(0.9f))};
  std::vector<int> shape = {2, 2};

  Tensor *input = make_tensor(data, shape);
  Tensor *result = input->atanh();
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "atanh() failed";
  delete input;
  delete result;
  delete expected;
}
