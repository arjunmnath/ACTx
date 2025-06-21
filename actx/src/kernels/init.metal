#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

kernel void __ones__(device float *A [[buffer(0)]],
                     constant int *metadata [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  A[tid] = 1;
}

kernel void __full__(constant float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = input[0];
}

kernel void __eye__(device float *A [[buffer(0)]],
                    constant int *metadata [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  uint n = metadata[2]; // 0 -> total elements, 1-> rank(A), 2->n
  uint row = tid / n;
  uint col = tid % n;
  if (row < n && col < n) {
    A[tid] = (row == col) ? 1.0f : 0.0f;
  }
}
kernel void __zeros__(device float *A [[buffer(0)]],
                      constant int *metadata [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  A[tid] = 0;
}
