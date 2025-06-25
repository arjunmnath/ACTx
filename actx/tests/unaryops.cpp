#include "tensor.h"
#include "utility.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorUnary, Sqrt) {
  std::vector<float> data = {4.0, 9.0, 16.0, 25.0};
  std::vector<float> expected_data = {2.0, 3.0, 4.0, 5.0};
  std::vector<int> shape = {2, 2};

  Tensor *t = make_tensor(data, shape);
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(t->sqrt()->logical_e(expected)->all()) << "Sqrt failed";
}

TEST(TensorUnary, Exp) {
  std::vector<float> data = {0.0, 1.0, 2.0, 3.0};
  std::vector<float> expected_data = {std::exp(0.0f), std::exp(1.0f),
                                      std::exp(2.0f), std::exp(3.0f)};
  std::vector<int> shape = {2, 2};

  Tensor *t = make_tensor(data, shape);
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(t->exp()->logical_e(expected)->all()) << "Exp failed";
}

TEST(TensorUnary, Log) {
  std::vector<float> data = {1.0, std::exp(1.0f), std::exp(2.0f),
                             std::exp(3.0f)};
  std::vector<float> expected_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> shape = {2, 2};

  Tensor *t = make_tensor(data, shape);
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(t->log()->logical_e(expected)->all()) << "Log failed";
}

TEST(TensorUnary, Log10) {
  std::vector<float> data = {1.0, 10.0, 100.0, 1000.0};
  std::vector<float> expected_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> shape = {2, 2};

  Tensor *t = make_tensor(data, shape);
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(t->log10()->logical_e(expected)->all()) << "Log10 failed";
}
TEST(TensorUnary, Log2) {
  std::vector<float> data = {1.0, 2.0, 4.0, 8.0};
  std::vector<float> expected_data = {0.0, 1.0, 2.0, 3.0};
  std::vector<int> shape = {2, 2};

  Tensor *t = make_tensor(data, shape);
  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(t->log2()->logical_e(expected)->all()) << "Log2 failed";
}
TEST(TensorUnary, LogHandlesNegativeAndZero) {
  std::vector<float> data = {-1.0, 0.0, 1.0, 2.0, INFINITY, -INFINITY};
  std::vector<int> shape = {2, 3};
  Tensor *t = make_tensor(data, shape);
  Tensor *result = nullptr;
  EXPECT_NO_THROW({ result = t->log(); });
  EXPECT_TRUE(std::isnan(result->_get_element(0))) << "log(-1) should be nan";
  EXPECT_TRUE(std::isinf(result->_get_element(1)) &&
              result->_get_element(1) < 0)
      << "log(0) should be -inf";
  EXPECT_NEAR(result->_get_element(2), 0.0f, 1e-6) << "log(1) should be 0";
  EXPECT_NEAR(result->_get_element(3), std::log(2.0f), 1e-6)
      << "log(2) should match";
  delete t;
  delete result;
}
