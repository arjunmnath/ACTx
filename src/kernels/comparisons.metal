#include <metal_stdlib>
using namespace metal;

kernel void logical_e(device float *A [[buffer(0)]],
                      device float *B [[buffer(1)]],
                      device float *C [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  C[tid] = A[tid] == B[tid];
}

kernel void logical_ne(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {

  C[tid] = A[tid] != B[tid];
}

kernel void logical_gt(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {

  C[tid] = A[tid] > B[tid];
}

kernel void logical_gte(device float *A [[buffer(0)]],
                        device float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {

  C[tid] = A[tid] >= B[tid];
}

kernel void logical_lt(device float *A [[buffer(0)]],
                       device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
  C[tid] = A[tid] < B[tid];
}

kernel void logical_lte(device float *A [[buffer(0)]],
                        device float *B [[buffer(1)]],
                        device float *C [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {

  C[tid] = A[tid] <= B[tid];
}
