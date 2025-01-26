#include <metal_stdlib>
using namespace metal;

kernel void __ones__(device float *A [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {

  A[tid] = 1;

}
kernel void __full__(device float *A [[buffer(0)]],
                      constant uint &value [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {

    A[tid] = value;
}
kernel void __eye__(device float *A [[buffer(0)]],
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
kernel void __zeros__(device float *A [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {

    A[tid] = 0;
}
