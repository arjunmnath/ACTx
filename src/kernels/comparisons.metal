#include <metal_stdlib>
using namespace metal;

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


