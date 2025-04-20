#pragma once

#include "memory.h"
#include "types.h"
#include <memory>
#include <set>

class MemoryPool {
private:
  std::multiset<std::shared_ptr<Memory>> available_pool;
  std::multiset<std::shared_ptr<Memory>> used_pool;
  size_t _compute_pool_size(size_t requested_size);

public:
  std::shared_ptr<Memory> request_memory(DeviceType device, size_t size,
                                         DType dtype);
  std::shared_ptr<Memory> find_suitable_block(size_t requested);
  void return_memory(std::shared_ptr<Memory> memory);
};
