#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void __pow__(device float *A [[buffer(0)]],
                    device float *B [[buffer(1)]],
                    device float *C [[buffer(2)]],
                    constant int *metadata [[buffer(3)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  C[tid] = pow(A[tid], B[0]);
}

kernel void __exp__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  output[tid] = exp(input[tid]);
}

kernel void __log__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  output[tid] = log(input[tid]);
}

kernel void __log10__(device float *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant int *metadata [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  output[tid] = log10(input[tid]);
}

kernel void __log2__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  output[tid] = log2(input[tid]);
}
kernel void __sqrt__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
  if ((int)tid >= metadata[0])
    return;
  output[tid] = sqrt(input[tid]);
}
