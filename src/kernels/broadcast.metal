#include <metal_stdlib>
using namespace metal;

inline int compute_broadcast_index(int flat_index, constant int *source_shape,
                                   constant int *target_shape, int source_rank,
                                   int target_rank) {

  int source_index = 0;
  int stride = 1;

  for (int i = target_rank - 1; i >= 0; --i) {
    int target_dim = target_shape[i];
    int coord = (flat_index % target_dim);

    if (i >= target_rank - source_rank) {
      int source_dim = source_shape[i - (target_rank - source_rank)];
      if (source_dim > 1) {
        source_index += coord * stride;
      }
    }

    flat_index /= target_dim;
    if (i >= target_rank - source_rank) {
      stride *= (source_shape[i - (target_rank - source_rank)] > 1
                     ? source_shape[i - (target_rank - source_rank)]
                     : 1);
    }
  }
  return source_index;
}

kernel void add_broadcast(
    device float *A [[buffer(0)]], device float *B [[buffer(1)]],
    device float *C [[buffer(2)]], constant int *lshape [[buffer(3)]],
    constant int *rshape [[buffer(4)]], constant int *target [[buffer(5)]],
    constant int *ranks [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  int flat_index = tid;
  int lrank = ranks[0];
  int rrank = ranks[1];
  int trank = ranks[2];
  int lindex =
      compute_broadcast_index(flat_index, lshape, target, lrank, trank);
  int rindex =
      compute_broadcast_index(flat_index, rshape, target, rrank, trank);
  C[flat_index] = A[lindex] + B[rindex];
}
