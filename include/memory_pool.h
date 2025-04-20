#pragma once

#include "memory.h"
#include "types.h"
#include <memory>
#include <set>
struct MemoryComparator {
  bool operator()(const std::shared_ptr<Memory> &a,
                  const std::shared_ptr<Memory> &b) const {
    return a->size < b->size;
  }
};
class MemoryPool {
private:
  std::multiset<std::shared_ptr<Memory>, MemoryComparator> available_pool;
  std::multiset<std::shared_ptr<Memory>, MemoryComparator> used_pool;
  size_t _compute_pool_size(size_t requested_size);

public:
  std::shared_ptr<Memory> request_memory(DeviceType device, size_t size,
                                         DType dtype);
  std::shared_ptr<Memory> find_suitable_block(DeviceType device, DType dtype,
                                              size_t requested);
  void return_memory(std::shared_ptr<Memory> memory);
};
