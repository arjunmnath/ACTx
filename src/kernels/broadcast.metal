
inline int compute_broadcast_index(int  flat_index,
                                   constant int *source_shape,
                                   constant int *target_shape,
                                   int  source_rank,
                                   int  target_rank) {
  bool shapes_match = true;
  for (int d = 0; d < target_rank; ++d) {
    if (source_shape[d] != target_shape[d]) {
      shapes_match = false;
      break;
    }
  }
  if (shapes_match) {
      return flat_index;
  }

  int source_index = 0;
  int stride       = 1;

  for (int i = target_rank - 1; i >= 0; --i) {
    int target_dim = target_shape[i];
    int coord      = flat_index % target_dim;

    if (i >= target_rank - source_rank) {
      int source_dim = source_shape[i - (target_rank - source_rank)];
      if (source_dim > 1) {
        source_index += coord * stride;
      }
    }

    flat_index /= target_dim;
    if (i >= target_rank - source_rank) {
      int source_dim = source_shape[i - (target_rank - source_rank)];
      stride *= (source_dim > 1 ? source_dim : 1);
    }
  }

  return source_index;
}

