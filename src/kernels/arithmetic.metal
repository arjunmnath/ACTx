#include <metal_stdlib>
#include "./broadcast.metal"
using namespace metal;
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


