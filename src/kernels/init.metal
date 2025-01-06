#include <metal_stdlib>
using namespace metal;

kernel void init_ones(device float *A [[buffer(0)]],
                      constant uint2 &dims [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    A[row * N + col] = 1;
  }
}
kernel void init_full(device float *A [[buffer(0)]],
                      constant uint &value [[buffer(1)]],
                      constant uint2 &dims [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    A[row * N + col] = value;
  }
}
kernel void init_identity(device float *A [[buffer(0)]],
                          constant uint2 &dims [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {

  uint m = dims.x;
  uint n = dims.y;
  uint row = tid / n;
  uint col = tid % n;
  if (row < m && col < n && row == col) {
    A[row * n + col] = 1;
  }
}
kernel void init_with_zeros(device float *A [[buffer(0)]],
                            constant uint2 &dims [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    A[row * N + col] = 0;
  }
}
