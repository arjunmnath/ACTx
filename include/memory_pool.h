#pragma once

#include "memory.h"
#include "types.h"
#include <memory>
#include <set>
#include <vector>

class MemoryPool {
private:
  std::multiset<std::shared_ptr<Memory>> available_pool;
  std::multiset<std::shared_ptr<Memory>> used_pool;
  int _compute_pool_size(int requested_size);

public:
  std::shared_ptr<Memory> request_memory(DeviceType device, int size,
                                         DType dtype);
  std::shared_ptr<Memory> find_suitable_block(int requested);
  void return_memory(std::shared_ptr<Memory> memory);
};
