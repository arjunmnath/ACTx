#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void __ones__(device float *A [[buffer(0)]],
                     constant uint2 &meta [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {

  if (tid >= meta.x * meta.y)
    return;
  A[tid] = 1;
}
kernel void __full__(device float *A [[buffer(0)]],
                     constant float &value [[buffer(1)]],
                     constant uint2 &meta [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  uint total_elems = meta[0];
  if (tid >= meta.x * meta.y)
    return;
  A[tid] = value;
}
kernel void __eye__(device float *A [[buffer(0)]],
                    constant uint2 &dims [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {

  uint m = dims.x;
  uint n = dims.y;
  uint row = tid / n;
  uint col = tid % n;
    if (row < n && col < n) {
        A[tid] = (row == col) ? 1.0f : 0.0f;
    }
}
kernel void __zeros__(device float *A [[buffer(0)]],
                      constant uint2 &meta [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {

  if (tid >= meta.x * meta.y)
    return;
  A[tid] = 0;
}
