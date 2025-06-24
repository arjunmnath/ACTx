#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void __sin__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = sin(input[tid]);
}

kernel void __cos__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = cos(input[tid]);
}

kernel void __tan__(device float *input [[buffer(0)]],
                    device float *output [[buffer(1)]],
                    constant int *metadata [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = tan(input[tid]);
}

kernel void __asin__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = asin(input[tid]);
}
kernel void __acos__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = acos(input[tid]);
}
kernel void __atan__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = atan(input[tid]);
}

kernel void __atan2__(device const float *A [[buffer(0)]],
                      device const float *B [[buffer(1)]],
                      device float *C [[buffer(2)]],
                      constant int *metadata [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;

  int arank = metadata[1];
  int brank = metadata[2];
  int rrank = metadata[3];
  const constant int *ashape = metadata + 4;
  const constant int *astride = ashape + arank;
  const constant int *bshape = astride + arank;
  const constant int *bstride = bshape + brank;
  const constant int *result_shape = bstride + brank;
  int ai =
      compute_broadcast_index(tid, ashape, astride, result_shape, arank, rrank);
  int bi =
      compute_broadcast_index(tid, bshape, bstride, result_shape, brank, rrank);
  C[tid] = atan2(B[bi], A[ai]);
}

kernel void __sinh__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = sinh(input[tid]);
}

kernel void __cosh__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = cosh(input[tid]);
}

kernel void __tanh__(device float *input [[buffer(0)]],
                     device float *output [[buffer(1)]],
                     constant int *metadata [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = tanh(input[tid]);
}

kernel void __asinh__(device float *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant int *metadata [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = asinh(input[tid]);
}

kernel void __acosh__(device float *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant int *metadata [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = acosh(input[tid]);
}

kernel void __atanh__(device float *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant int *metadata [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {

  if ((int)tid >= metadata[0])
    return;
  output[tid] = atanh(input[tid]);
}
