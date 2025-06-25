#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

/*
TEST(TensorView, InPlaceAddModifiesOnlyView) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  Tensor *parent = new Tensor(data, {2, 3});

  // View second column: shape [2, 1]
  std::vector<Slice> view_slice = {Slice(0, 2), Slice(1, 2)};
  Tensor *view = parent->view(view_slice);

  Tensor *adder = new Tensor({10, 20}, {2, 1});
  view->add(adder, true); // In-place add

  // Expected: only column 1 modified
  std::vector<float> expected_data = {1, 12, 3, 4, 25, 6};
  EXPECT_EQ(parent->get_data(), expected_data)
      << "In-place add on view modified data outside the view.";

  delete parent;
  delete view;
  delete adder;
}
TEST(TensorView, MulDoesNotAffectParentIfNotInplace) {
  std::vector<float> data = {2, 4, 6, 8};
  Tensor *parent = new Tensor(data, {2, 2});

  std::vector<Slice> view_slice = {Slice(0, 2), Slice(0, 1)};
  Tensor *view = parent->view(view_slice); // first column
  std::vector<float> data2 = {2, 3};
  Tensor *mul = new Tensor(data2, {2, 1});
  Tensor *result = view->mul(mul, false); // not in-place

  std::vector<float> expected_result = {4, 12};
  Tensor *expected = new Tensor(expected_result, {2, 1});
  EXPECT_TRUE(result->logical_e(expected)->all())
      << "Mul result on view is incorrect.";

  // Parent should remain untouched
  std::vector<float> expected_parent = {2, 4, 6, 8};

  Tensor *expectedp = new Tensor(expected_parent, {2, 2});
  EXPECT_TRUE(parent->logical_e(expectedp)->all())
      << "Non-inplace mul on view altered the parent.";

  delete parent;
  delete view;
  delete mul;
  delete result;
}
TEST(TensorView, SubtractionFromViewWorksCorrectly) {
  std::vector<float> data = {10, 20, 30, 40};
  Tensor *parent = new Tensor(data, {2, 2});

  std::vector<Slice> slice = {Slice(0, 2), Slice(1, 2)}; // Second column
  Tensor *view = parent->view(slice);

  Tensor *sub_tensor = new Tensor({5, 15}, {2, 1});
  Tensor *result = view->sub(sub_tensor, false);

  Tensor *expected = new Tensor({15, 25}, {2, 1});
  EXPECT_TRUE(result->logical_e(expected)->all())
      << "View subtraction returned incorrect result.";

  delete parent;
  delete view;
  delete sub_tensor;
  delete result;
  delete expected;
}

TEST(TensorView, DivisionOnViewIsLocalised) {
  std::vector<float> data = {8, 16, 24, 32};
  Tensor *parent = new Tensor(data, {2, 2});

  std::vector<Slice> slice = {Slice(0, 2), Slice(1, 2)}; // Second column
  Tensor *view = parent->view(slice);

  Tensor *div_tensor = new Tensor({2, 4}, {2, 1});
  view->div(div_tensor, true); // in-place

  std::vector<float> expected_parent = {8, 8, 24, 8};

  EXPECT_EQ(parent->get_data(), expected_parent)
      << "In-place division on view affected wrong elements.";

  delete parent;
  delete view;
  delete div_tensor;
}*/
