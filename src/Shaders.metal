#include <metal_stdlib>
using namespace metal;

// =====================================================================
//                          ARITHMETIC
// =====================================================================
kernel void add_matrix(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant uint2 &dims [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] + B[row * N + col];
  }
}

kernel void subtract_matrix(device float *A [[buffer(0)]],
                            device float *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant uint2 &dims
                            [[buffer(3)]], // buffer for M, N, P
                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] - B[row * N + col];
  }
}

kernel void elementwise_divide_matrix(device float *A [[buffer(0)]],
                                      device float *B [[buffer(1)]],
                                      device float *C [[buffer(2)]],
                                      constant uint2 &dims [[buffer(3)]],
                                      uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] / B[row * N + col];
  }
}

kernel void elementwise_multiply_matrix(device float *A [[buffer(0)]],
                                        device float *B [[buffer(1)]],
                                        device float *C [[buffer(2)]],
                                        constant uint2 &dims [[buffer(3)]],
                                        uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] * B[row * N + col];
  }
}

kernel void elementwise_pow(device float *A [[buffer(0)]],
                            device float *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant uint2 &dims [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = pow(A[row * N + col], B[0]);
  }
}

kernel void matrix_multiply(device float *A [[buffer(0)]],
                            device float *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant uint3 &dims [[buffer(3)]],
                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint P = dims.z;
  uint row = tid / P;
  uint col = tid % P;

  if (row < M && col < P) {
    float sum = 0.0;
    for (uint k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * P + col];
    }
    C[row * P + col] = sum;
  }
}
// =====================================================================

// =====================================================================
//                          COMPARISIONS
// =====================================================================
kernel void logical_e(device float *A [[buffer(0)]],
                      device float *B [[buffer(1)]],
                      device float *C [[buffer(2)]],
                      constant uint2 &dims [[buffer(3)]], // buffer for M, N, P
                      uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] == B[row * N + col];
  }
}

kernel void logical_ne(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant uint2 &dims [[buffer(3)]], // buffer for M, N, P
                       uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] != B[row * N + col];
  }
}

kernel void logical_gt(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant uint2 &dims [[buffer(3)]], // buffer for M, N, P
                       uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] > B[row * N + col];
  }
}

kernel void logical_gte(device float *A [[buffer(0)]],
                        device float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        constant uint2 &dims
                        [[buffer(3)]], // buffer for M, N, P
                        uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] >= B[row * N + col];
  }
}

kernel void logical_lt(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       constant uint2 &dims [[buffer(3)]], // buffer for M, N, P
                       uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] < B[row * N + col];
  }
}

kernel void logical_lte(device float *A [[buffer(0)]],
                        device float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        constant uint2 &dims
                        [[buffer(3)]], // buffer for M, N, P
                        uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = A[row * N + col] <= B[row * N + col];
  }
}

// =====================================================================

// =====================================================================
//                            MATH
// =====================================================================
kernel void exp(device float *A [[buffer(0)]], device float *C [[buffer(1)]],
                constant uint2 &dims [[buffer(2)]],
                uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = exp(A[row * N + col]);
  }
}

kernel void log(device float *A [[buffer(0)]], device float *C [[buffer(1)]],
                constant uint2 &dims [[buffer(2)]],
                uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = log(A[row * N + col]);
  }
}
// =====================================================================
