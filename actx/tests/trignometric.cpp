#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorTrig, SinOperation) {
  std::vector<float> data = {0.0, M_PI_2, M_PI, 3 * M_PI_2};
  std::vector<float> expected_data = {0.0, 1.0, 0.0, -1.0};
  std::vector<int> shape = {2, 2};

  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->sin(false);
  Tensor *expected = new Tensor(expected_data, shape);
  EXPECT_TRUE(result->logical_e(expected)->all()) << "sin() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, CosOperation) {
  std::vector<float> data = {0.0, M_PI_2, M_PI, 3 * M_PI_2};
  std::vector<float> expected_data = {1.0, 0.0, -1.0, 0.0};
  std::vector<int> shape = {2, 2};

  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->cos();
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "cos() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, TanOperation) {
  std::vector<float> data = {0.0, M_PI_4, -M_PI_4, M_PI};
  std::vector<float> expected_data = {0.0, 1.0, -1.0, 0.0};
  std::vector<int> shape = {2, 2};

  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->tan();
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "tan() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AsinOperation) {
  std::vector<float> data = {0.0, 0.5, -0.5, 1.0};
  std::vector<float> expected_data = {0.0, static_cast<float>(std::asin(0.5)),
                                      static_cast<float>(std::asin(-0.5)),
                                      M_PI_2};
  std::vector<int> shape = {2, 2};
  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->asin();
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "asin() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AcosOperation) {
  std::vector<float> data = {1.0, 0.5, -0.5, -1.0};
  std::vector<float> expected_data = {0.0, static_cast<float>(std::acos(0.5)),
                                      static_cast<float>(std::acos(-0.5)),
                                      M_PI};
  std::vector<int> shape = {2, 2};
  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->acos();
  Tensor *expected = new Tensor(expected_data, shape);
  EXPECT_TRUE(result->logical_e(expected)->all()) << "acos() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, AtanOperation) {
  std::vector<float> data = {0.0, 1.0, -1.0, 10.0};
  std::vector<float> expected_data = {0.0, static_cast<float>(std::atan(1.0)),
                                      static_cast<float>(std::atan(-1.0)),
                                      static_cast<float>(std::atan(10.0))};
  std::vector<int> shape = {2, 2};

  Tensor *input = new Tensor(data, shape);
  Tensor *result = input->atan();
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "atan() failed";
  delete input;
  delete result;
  delete expected;
}

TEST(TensorTrig, Atan2Operation) {
  std::vector<float> y_data = {0.0, 1.0, -1.0, 1.0};
  std::vector<float> x_data = {1.0, 1.0, 1.0, -1.0};
  std::vector<float> expected_data = {
      static_cast<float>(std::atan2(0.0, 1.0)),
      static_cast<float>(std::atan2(1.0, 1.0)),
      static_cast<float>(std::atan2(-1.0, 1.0)),
      static_cast<float>(std::atan2(1.0, -1.0))};
  std::vector<int> shape = {2, 2};

  Tensor *y = new Tensor(y_data, shape);
  Tensor *x = new Tensor(x_data, shape);
  Tensor *result = x->atan2(y);
  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(result->logical_e(expected)->all()) << "atan2() failed";
  delete x;
  delete y;
  delete result;
  delete expected;
}
