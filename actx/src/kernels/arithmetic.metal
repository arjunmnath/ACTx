#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void
__add__(device const float *A [[buffer(0)]],
        device const float *B [[buffer(1)]], device float *C [[buffer(2)]],
        constant int *lshape [[buffer(3)]], constant int *rshape [[buffer(4)]],
        constant int *outshape [[buffer(5)]], constant int *ranks [[buffer(6)]],
        uint tid [[thread_position_in_grid]]) {
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];

  int li = compute_broadcast_index(tid, lshape, outshape, lrank, trank);
  int ri = compute_broadcast_index(tid, rshape, outshape, rrank, trank);

  C[tid] = A[li] + B[ri];
}
kernel void __sub__(device float *A [[buffer(0)]],
                    device float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *lshape [[buffer(3)]],
                    constant int *rshape [[buffer(4)]],
                    constant int *result_shape [[buffer(5)]],
                    constant int *ranks [[buffer(6)]],
                    uint tid [[thread_position_in_grid]]) {
  int flat_index = tid;
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];
  int lindex =
      compute_broadcast_index(flat_index, lshape, result_shape, lrank, trank);
  int rindex =
      compute_broadcast_index(flat_index, rshape, result_shape, rrank, trank);
  C[flat_index] = A[lindex] - B[rindex];
}

kernel void __div__(device float *A [[buffer(0)]],
                    device float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *lshape [[buffer(3)]],
                    constant int *rshape [[buffer(4)]],
                    constant int *result_shape [[buffer(5)]],
                    constant int *ranks [[buffer(6)]],
                    uint tid [[thread_position_in_grid]]) {
  int flat_index = tid;
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];
  int lindex =
      compute_broadcast_index(flat_index, lshape, result_shape, lrank, trank);
  int rindex =
      compute_broadcast_index(flat_index, rshape, result_shape, rrank, trank);
  C[flat_index] = A[lindex] / B[rindex];
}

kernel void __mul__(device float *A [[buffer(0)]],
                    device float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *lshape [[buffer(3)]],
                    constant int *rshape [[buffer(4)]],
                    constant int *result_shape [[buffer(5)]],
                    constant int *ranks [[buffer(6)]],
                    uint tid [[thread_position_in_grid]]) {
  int flat_index = tid;
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];
  int lindex =
      compute_broadcast_index(flat_index, lshape, result_shape, lrank, trank);
  int rindex =
      compute_broadcast_index(flat_index, rshape, result_shape, rrank, trank);
  C[flat_index] = A[lindex] * B[rindex];
}

// FIX: matmul algorithm to match n dimensional tensors
kernel void __matmul__(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant int *lshape [[buffer(3)]],
                       constant int *rshape [[buffer(4)]],
                       constant int *result_shape [[buffer(5)]],
                       constant int *ranks [[buffer(6)]],
                       uint tid [[thread_position_in_grid]]) {
  int flat_index = tid;
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];
  int lindex =
      compute_broadcast_index(flat_index, lshape, result_shape, lrank, trank);
  int rindex =
      compute_broadcast_index(flat_index, rshape, result_shape, rrank, trank);
  C[flat_index] = A[lindex] * B[rindex];
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
