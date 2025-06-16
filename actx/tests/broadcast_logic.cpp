#include <gtest/gtest.h>
#include <vector>

int compute_broadcast_index(int flat_index,
                            const std::vector<int> &source_shape,
                            const std::vector<int> &target_shape) {
  int source_rank = source_shape.size();
  int target_rank = target_shape.size();
  int offset = target_rank - source_rank;

  int source_index = 0;
  int stride = 1;
  int idx = flat_index;

  for (int i = target_rank - 1; i >= 0; --i) {
    int target_dim = target_shape[i];
    int coord = idx % target_dim;
    idx /= target_dim;

    int src_dim = 1;
    if (i - offset >= 0)
      src_dim = source_shape[i - offset];

    int src_coord = (src_dim == 1) ? 0 : coord;
    source_index += src_coord * stride;
    stride *= src_dim;
  }

  return source_index;
}

std::vector<float> broadcast_add(const std::vector<float> &A,
                                 const std::vector<int> &ashape,
                                 const std::vector<float> &B,
                                 const std::vector<int> &bshape,
                                 const std::vector<int> &outshape) {
  int total = 1;
  for (int d : outshape)
    total *= d;

  std::vector<float> C(total);

  for (int i = 0; i < total; ++i) {
    int ai = compute_broadcast_index(i, ashape, outshape);
    int bi = compute_broadcast_index(i, bshape, outshape);
    C[i] = A[ai] + B[bi];
  }

  return C;
}

class BroadcastAddTest : public ::testing::Test {};

TEST_F(BroadcastAddTest, ScalarBroadcast) {
  std::vector<float> A(12, 1.0); // shape [3,4]
  std::vector<float> B = {1.0};  // shape [1]
  std::vector<float> expected(12, 2.0);
  auto result = broadcast_add(A, {3, 4}, B, {1}, {3, 4});
  EXPECT_EQ(result, expected);
}

TEST_F(BroadcastAddTest, VectorBroadcast) {
  std::vector<float> A = {1, 2, 3, 4, 5, 6}; // shape [2,3]
  std::vector<float> B = {10, 20, 30};       // shape [3]
  std::vector<float> expected = {11, 22, 33, 14, 25, 36};
  auto result = broadcast_add(A, {2, 3}, B, {3}, {2, 3});
  EXPECT_EQ(result, expected);
}

TEST_F(BroadcastAddTest, ExpandDimsBroadcast) {
  std::vector<float> A = {5, 10};                      // shape [2,1]
  std::vector<float> B = {1, 2, 3, 4, 5, 6};           // shape [2,3]
  std::vector<float> expected = {6, 7, 8, 14, 15, 16}; // <-- fixed here
  auto result = broadcast_add(A, {2, 1}, B, {2, 3}, {2, 3});
  EXPECT_EQ(result, expected);
}
TEST_F(BroadcastAddTest, Case1) {
  std::vector<float> A = {1, 1, 1, 1, 1, 1, 1, 1, 1};        // shape [2,1]
  std::vector<float> B = {1};                                // shape [2,3]
  std::vector<float> expected = {2, 2, 2, 2, 2, 2, 2, 2, 2}; // <-- fixed here
  auto result = broadcast_add(A, {3, 3}, B, {1}, {3, 3});
  EXPECT_EQ(result, expected);
}
