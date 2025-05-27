
inline int compute_broadcast_index(int flat_index, constant int *source_shape,
                                   constant int *target_shape, int source_rank,
                                   int target_rank) {
  int source_index = 0;
  int stride = 1;

  for (int i = target_rank - 1; i >= 0; --i) {
    int target_dim = target_shape[i];
    int coord = flat_index % target_dim;

    int src_i = i - (target_rank - source_rank);

    if (src_i >= 0) {
      int source_dim = source_shape[src_i];
      if (source_dim > 1) {
        source_index += coord * stride;
        stride *= source_dim;
      } else {
        stride *= 1;
      }
    }

    flat_index /= target_dim;
  }

  return source_index;
}
