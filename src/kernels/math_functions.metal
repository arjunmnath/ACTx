#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void __pow__(device float *A [[buffer(0)]],
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

kernel void sqrt(device float *A [[buffer(0)]], device float *C [[buffer(1)]],
                 constant uint2 &dims [[buffer(2)]],
                 uint tid [[thread_position_in_grid]]) {
  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    C[row * N + col] = sqrt(A[row * N + col]);
  }
}
