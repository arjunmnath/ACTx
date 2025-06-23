inline int compute_broadcast_index(uint tid, constant int *source_shape,
                                   constant int *source_stride,
                                   constant int *target_shape, int source_rank,
                                   int target_rank) {
  int offset = target_rank - source_rank;
  int source_index = 0;

  for (int i = target_rank - 1; i >= 0; --i) {
    int coord = tid % target_shape[i];
    tid /= target_shape[i];

    if (i - offset >= 0) {
      int src_dim = source_shape[i - offset];
      int src_stride = source_stride[i - offset];
      int src_coord = (src_dim == 1) ? 0 : coord;
      source_index += src_coord * src_stride;
    }
  }
  return source_index;
}
