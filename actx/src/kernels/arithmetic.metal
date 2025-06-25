#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void __add__(device const float *A [[buffer(0)]],
                    device const float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *metadata [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;

  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;
  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = A[ai] + B[bi];
}

kernel void __sub__(device const float *A [[buffer(0)]],
                    device const float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *metadata [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;

  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = A[ai] - B[bi];
}
kernel void __div__(device const float *A [[buffer(0)]],
                    device const float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *metadata [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;

  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = A[ai] / B[bi];
}

kernel void __mul__(device const float *A [[buffer(0)]],
                    device const float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *metadata [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;
  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = A[ai] * B[bi];
}

// FIX: matmul algorithm to match n dimensional tensors
kernel void __matmul__(device const float *A [[buffer(0)]],
                       device const float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant int *metadata [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;
  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = A[ai] * B[bi];
}

kernel void __neg__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = input[tid] * -1.0f;
}

/*
kernel void tensor_matrix_multiply(
    device float *A [[buffer(0)]],       // Left tensor
    device float *B [[buffer(1)]],       // Right tensor
    device float *C [[buffer(2)]],       // Output tensor
    constant int *A_shape [[buffer(3)]], // Shape of left tensor
    constant int *B_shape [[buffer(4)]], // Shape of right tensor
    constant int *C_shape [[buffer(5)]], // Shape of output tensor
    constant int rank [[buffer(6)]],     // Tensor rank
    uint3 grid_pos [[thread_position_in_grid]]) {
  // Compute global output index
  int global_output_index = 0;
  for (int i = 0; i < rank; i++) {
    global_output_index += grid_pos[i] * C_shape[i];
  }

  // Initialize result
  float result = 0.0f;

  // Hardcoded matrix multiplication for last two dimensions
  for (int k = 0; k < A_shape[rank - 1]; k++) {
    // Compute source indices for A and B
    int A_index = 0, B_index = 0;
    for (int i = 0; i < rank; i++) {
      if (i == rank - 2) {
        A_index += grid_pos[i] * A_shape[i];
        B_index += grid_pos[i] * B_shape[i];
      } else if (i == rank - 1) {
        A_index += k;
        B_index += k * B_shape[i];
      } else {
        A_index += grid_pos[i] * A_shape[i];
        B_index += grid_pos[i] * B_shape[i];
      }
    }
    // Multiply and accumulate
    result += A[A_index] * B[B_index];
  }

  // Store result
  C[global_output_index] = result;
}
*/
