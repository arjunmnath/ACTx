
#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

TEST(TensorSlice, BasicSliceWorks) {
  std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<int> shape = {10};
  std::vector<Slice> slices = {Slice(2, 7, 1)};
  Tensor *tensor = new Tensor(data, shape);

  // Slice [2:7]
  Tensor *result = tensor->view(slices);

  std::vector<float> expected_data = {2, 3, 4, 5, 6};

  Tensor *expected = new Tensor(expected_data, {5});

  // Check that the sliced tensor is equal to the expected one
  EXPECT_TRUE(result->logical_e(expected)->all()) << "Tensor slicing failed";

  delete tensor;
  delete result;
  delete expected;
}

TEST(TensorSlice, SliceWithStepWorks) {
  std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<int> shape = {10};
  std::vector<Slice> slices = {Slice(1, 8, 2)};
  Tensor *tensor = new Tensor(data, shape);

  // Slice [1:8:2]
  Tensor *result = tensor->view(slices);

  std::vector<float> expected_data = {1, 3, 5, 7};

  Tensor *expected = new Tensor(expected_data, {4});

  // Check that the sliced tensor is equal to the expected one
  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor slicing with step failed";

  delete tensor;
  delete result;
  delete expected;
}

TEST(TensorSlice, NegativeIndexWorks) {
  std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7};

  std::vector<int> shape = {8};
  std::vector<Slice> slices = {Slice(-5, -1, 1)};
  Tensor *tensor = new Tensor(data, shape);

  // Slice [-5:-1]
  Tensor *result = tensor->view(slices);

  std::vector<float> expected_data = {3, 4, 5, 6};

  Tensor *expected = new Tensor(expected_data, {4});

  // Check that the sliced tensor is equal to the expected one
  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Tensor slicing with negative indices failed";

  delete tensor;
  delete result;
  delete expected;
}

TEST(TensorSlice, Slice2DWorks) {
  // 2D array 3x4
  std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int> shape = {3, 4};
  std::vector<Slice> slices = {Slice(1, 3, 1), Slice(1, 4, 2)};
  Tensor *tensor = new Tensor(data, shape);

  // Slice [1:3, 1:4:2]
  Tensor *result = tensor->view(slices);

  // Expected:
  // row 1, col 1
  // row 1, col 3
  // row 2, col 1
  // row 2, col 3
  std::vector<float> expected_data = {5, 7, 9, 11};
  Tensor *expected = new Tensor(expected_data, {2, 2});

  // Check that the 2D sliced tensor is equal to the expected one
  EXPECT_TRUE(result->logical_e(expected)->all()) << "2D Tensor slicing failed";

  delete tensor;
  delete result;
  delete expected;
}
