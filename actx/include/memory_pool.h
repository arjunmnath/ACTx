#pragma once

#include "memory.h"
#include "types.h"
#include <memory>
#include <set>
struct MemoryComparator {
  bool operator()(const Memory *a, const Memory *b) const {
    return a->bytesize < b->bytesize;
  }
};
class MemoryPool {
private:
  std::multiset<Memory *, MemoryComparator> available_pool;
  std::multiset<Memory *, MemoryComparator> used_pool;
  size_t _compute_pool_size(size_t requested_size);

public:
  Memory *request_memory(DeviceType device, size_t length, DType dtype);
  Memory *find_suitable_block(DeviceType device, DType dtype, size_t requested);
  void return_memory(Memory *memory);
};
