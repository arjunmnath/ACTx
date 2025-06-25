#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

TEST(TensorLogicalOps, LogicalEqual) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {1, 2, 3, 4};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_e(b);
  std::vector<float> expected = {1, 1, 1, 1};
  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalOps, LogicalNotEqual) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {1, 0, 3, 0};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_ne(b);
  std::vector<float> expected = {0, 1, 0, 1};
  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalOps, LogicalGreaterThan) {
  std::vector<float> a_data = {1, 4, 3, 6};
  std::vector<float> b_data = {1, 2, 3, 5};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_gt(b);
  std::vector<float> expected = {0, 1, 0, 1};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalOps, LogicalGreaterThanOrEqual) {
  std::vector<float> a_data = {1, 4, 3, 6};
  std::vector<float> b_data = {1, 2, 3, 7};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_gte(b);
  std::vector<float> expected = {1, 1, 1, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalOps, LogicalLessThan) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {2, 2, 3, 5};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_lt(b);
  std::vector<float> expected = {1, 0, 0, 1};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalOps, LogicalLessThanOrEqual) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {2, 2, 3, 3};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, shape);
  Tensor *result = a->logical_lte(b);
  std::vector<float> expected = {1, 1, 1, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalEqualBroadcast) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {1};
  std::vector<int> a_shape = {2, 2};
  std::vector<int> b_shape = {1};

  Tensor *a = new Tensor(a_data, a_shape);
  Tensor *b = new Tensor(b_data, b_shape);
  Tensor *result = a->logical_e(b);
  std::vector<float> expected = {1, 0, 0, 0};

  Tensor *ex = new Tensor(expected, a_shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalNotEqualBroadcast) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {1};
  std::vector<int> a_shape = {2, 2};
  std::vector<int> b_shape = {1};

  Tensor *a = new Tensor(a_data, a_shape);
  Tensor *b = new Tensor(b_data, b_shape);
  Tensor *result = a->logical_ne(b);
  std::vector<float> expected = {0, 1, 1, 1};

  Tensor *ex = new Tensor(expected, a_shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalGreaterThanBroadcast) {
  std::vector<float> a_data = {5, 2, 3, 1};
  std::vector<float> b_data = {3};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, {1});
  Tensor *result = a->logical_gt(b);
  std::vector<float> expected = {1, 0, 0, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalGreaterThanOrEqualBroadcast) {
  std::vector<float> a_data = {3, 3, 4, 2};
  std::vector<float> b_data = {3};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, {1});
  Tensor *result = a->logical_gte(b);
  std::vector<float> expected = {1, 1, 1, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalLessThanBroadcast) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {3};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, {1});
  Tensor *result = a->logical_lt(b);
  std::vector<float> expected = {1, 1, 0, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}

TEST(TensorLogicalBroadcast, LogicalLessThanOrEqualBroadcast) {
  std::vector<float> a_data = {1, 2, 3, 4};
  std::vector<float> b_data = {3};
  std::vector<int> shape = {2, 2};

  Tensor *a = new Tensor(a_data, shape);
  Tensor *b = new Tensor(b_data, {1});
  Tensor *result = a->logical_lte(b);
  std::vector<float> expected = {1, 1, 1, 0};

  Tensor *ex = new Tensor(expected, shape);
  EXPECT_TRUE(result->logical_e(ex)->all());
}
