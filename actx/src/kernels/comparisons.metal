#include "./broadcast.metal"
#include <metal_stdlib>
using namespace metal;

kernel void logical_e(device const float *A [[buffer(0)]],
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
  float epsilon = 1e-5;
  C[tid] =
      fabs(A[ai] - B[bi]) < epsilon || (A[ai] == INFINITY && B[bi] == INFINITY)
          ? 1.0
          : 0.0;
}

kernel void logical_ne(device const float *A [[buffer(0)]],
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
  float epsilon = 1e-5;
  C[tid] = fabs(A[ai] - B[bi]) > epsilon ? 1.0 : 0.0;
}

kernel void logical_gt(device const float *A [[buffer(0)]],
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
  C[tid] = A[ai] > B[bi];
}

kernel void logical_gte(device const float *A [[buffer(0)]],
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
  C[tid] = A[ai] >= B[bi];
}

kernel void logical_lt(device const float *A [[buffer(0)]],
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
  C[tid] = A[ai] < B[bi];
}

kernel void logical_lte(device const float *A [[buffer(0)]],
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
  C[tid] = A[ai] <= B[bi];
}
