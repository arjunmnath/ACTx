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
// for (int i = target_rank - 1; i >= 0; --i) {
//   int target_dim = target_shape[i];
//   int coord = flat_index % target_dim;
//   int src_i = i - (target_rank - source_rank);
//   if (src_i >= 0) {
//     int source_dim = source_shape[src_i];
//     if (source_dim > 1) {
//       source_index += coord * stride;
//       stride *= source_dim;
//     } else {
//       stride *= 1;
//     }
//   }
//
//   flat_index /= target_dim;
// }
// return source_index;
// }
/*
int compute_broadcast_index(int flat_index,
                            const std::vector<int> &source_shape,
                            const std::vector<int> &target_shape,
                            const std::vector<int> &source_stride) {
  int source_rank = source_shape.size();
  int target_rank = target_shape.size();
  int offset = target_rank - source_rank;

  int source_index = 0;

  for (int i = target_rank - 1; i >= 0; --i) {
    int coord = flat_index % target_shape[i];
    flat_index /= target_shape[i];

    if (i - offset >= 0) {
      int src_dim = source_shape[i - offset];
      int src_stride = source_stride[i - offset];
      int src_coord = (src_dim == 1) ? 0 : coord;
      source_index += src_coord * src_stride;
    }
  }

  return source_index;
}
*/
