#include "tensor.h"
#include <gtest/gtest.h>

TEST(TensorInitalization, OnesNonSquare) {

  std::vector<int> shape = {3, 4};
  Tensor *tensor1 = Tensor::ones(shape);
  std::vector<float> expected_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  Tensor *expected = new Tensor(expected_data, shape);

  EXPECT_TRUE(tensor1->logical_e(expected)->all())
      << "Initalization of ones failed";
}

TEST(TensorInitalization, OnesSquare) {
  std::vector<int> shape = {2, 2};
  Tensor *a = Tensor::ones(shape);
  std::vector<float> ones = {1, 1, 1, 1};
  Tensor *expected = new Tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of ones failed";
}

TEST(TensorInitalization, ZerosNonSquare) {
  std::vector<int> shape = {2, 3};
  Tensor *a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0, 0, 0};
  Tensor *expected = new Tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of zeros failed";
}

TEST(TensorInitalization, ZerosSquare) {
  std::vector<int> shape = {2, 2};
  Tensor *a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0};
  Tensor *expected = new Tensor(ones, shape, DType::int32);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of zeros failed";
}

TEST(TensorInitalization, Eye) {
  std::vector<int> shape = {3, 3};
  Tensor *a = Tensor::eye(3);
  std::vector<float> ones = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  Tensor *expected = new Tensor(ones, shape);
  EXPECT_TRUE(a->logical_e(expected)->all()) << "Initalization of eye failed";
}
