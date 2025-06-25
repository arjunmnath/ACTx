#include "tensor.h"
#include "utility.h"
#include <gtest/gtest.h>

TEST(TensorInitalization, OnesNonSquare) {

  std::vector<int> shape = {3, 4};
  Tensor *tensor1 = Tensor::ones(shape);
  std::vector<float> expected_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  Tensor *expected = make_tensor(expected_data, shape);

  EXPECT_TRUE(tensor1->logical_e(expected)->all())
      << "Initalization of ones failed";
}

TEST(TensorInitalization, OnesSquare) {
  std::vector<int> shape = {2, 2};
  Tensor *a = Tensor::ones(shape);
  std::vector<float> ones = {1, 1, 1, 1};
  Tensor *expected = make_tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of ones failed";
}

TEST(TensorInitalization, ZerosNonSquare) {
  std::vector<int> shape = {2, 3};
  Tensor *a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0, 0, 0};
  Tensor *expected = make_tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of zeros failed";
}

TEST(TensorInitalization, ZerosSquare) {
  std::vector<int> shape = {2, 2};
  Tensor *a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0};
  Tensor *expected = make_tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of zeros failed";
}

TEST(TensorInitalization, Eye) {
  std::vector<int> shape = {3, 3};
  Tensor *a = Tensor::eye(3);
  std::vector<float> ones = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  Tensor *expected = make_tensor(ones, shape);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of eye failed";
}
TEST(TensorInitalization, FULL) {
  std::vector<int> shape = {3, 3};
  Tensor *a = Tensor::full(shape, 3.0f);
  std::vector<float> ones = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
                             3.0f, 3.0f, 3.0f, 3.0f};
  Tensor *expected = make_tensor(ones, shape);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of eye failed";
}

TEST(TensorInitalization, CLONE) {
  std::vector<int> shape = {3, 3};
  std::vector<float> data = {3.0f, -1.0f, 3.1f, 3.0f, 3.0f,
                             3.0f, 7.0f,  2.0f, 3.0f};
  Tensor *expected = make_tensor(data, shape);
  Tensor *copied = Tensor::clone(expected);
  EXPECT_TRUE(copied->logical_e(expected)->all())
      << "Initalization of eye failed";
}
