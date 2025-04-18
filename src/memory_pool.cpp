#include "memory_pool.h"
#include "memory.h"
#include <algorithm>
#include <cmath>
#include <memory>

int MemoryPool::_compute_pool_size(int requested_size) {
  /*
   * requested_size(11): 1011 -> 4 (block_size: 16)
   * requested_size(32): 100000 -> 5 (block_size: 32)
   */
  return pow(2, ceil(log2(requested_size)));
}

std::shared_ptr<Memory> MemoryPool::request_memory(DeviceType device, int size,
                                                   DType dtype) {
  int required_block_size = this->_compute_pool_size(size);
  std::shared_ptr<Memory> suitable_block =
      this->find_suitable_block(required_block_size);
  if (nullptr == suitable_block) {
    std::shared_ptr<Memory> memory =
        std::make_shared<Memory>(device, required_block_size, dtype);
    this->used_pool.insert(memory);
    return memory;
  }
  this->used_pool.insert(suitable_block);
  this->available_pool.erase(suitable_block);
  return suitable_block;
}

std::shared_ptr<Memory> MemoryPool::find_suitable_block(int requested) {
  auto it = std::upper_bound(this->available_pool.begin(),
                             this->available_pool.end(), requested,
                             [](int size, const std::shared_ptr<Memory> &mem) {
                               return size <= mem->size;
                             });

  if (it == this->available_pool.end()) {
    return nullptr;
  }
  std::shared_ptr<Memory> item = *it;
  int block_size = item->size;
  if (block_size >= requested * 2) {
    return nullptr;
  }
  return item;
}

void MemoryPool::return_memory(std::shared_ptr<Memory> memory) {
  this->used_pool.erase(memory);
  this->available_pool.insert(memory);
}
