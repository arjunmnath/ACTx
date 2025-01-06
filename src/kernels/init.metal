#include <metal_stdlib>
using namespace metal;


kernel void init_one(device float *A [[buffer(0)]], 
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

kernel void init_zero(device float *A [[buffer(0)]], 
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

kernel void init_full(device float *A [[buffer(0)]], 
                constant uint2 &dims [[buffer(1)]],
                constant uint &value[[buffer(2)]],
                uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;
  uint N = dims.y;
  uint row = tid / N;
  uint col = tid % N;
  if (row < M && col < N) {
    A[row * N + col] = value;
  }
}






