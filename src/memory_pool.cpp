#include "memory_pool.h"
#include "main.h"
#include "memory.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

size_t MemoryPool::_compute_pool_size(size_t requested_size) {
  /*
   * requested_size should be in bytes
   * requested_size(11): 1011 -> 4 (block_size: 16)
   * requested_size(32): 100000 -> 5 (block_size: 32)
   */
  return std::max(static_cast<int>(pow(2, ceil(log2(requested_size + 1)))), 2);
}

std::shared_ptr<Memory> MemoryPool::request_memory(DeviceType device,
                                                   size_t size, DType dtype) {

  int required_block_size = this->_compute_pool_size(size);
  std::shared_ptr<Memory> suitable_block =
      this->find_suitable_block(required_block_size);
  if (nullptr == suitable_block) {
    std::shared_ptr<Memory> memory =
        std::make_shared<Memory>(device, required_block_size, dtype);
    this->used_pool.insert(memory);
    logger->info("Requesting, Used Pool size: {} Available Pool Size: {} Pool "
                 "size: {} Requested Size: {}",
                 this->used_pool.size(), this->available_pool.size(),
                 required_block_size, size);
    return memory;
  }
  this->used_pool.insert(suitable_block);
  this->available_pool.erase(suitable_block);
  logger->info("Requesting, Used Pool size: {} Available Pool Size: {} Pool "
               "size: {} Requested Size: {}",
               this->used_pool.size(), this->available_pool.size(),
               required_block_size, size);
  return suitable_block;
}

std::shared_ptr<Memory> MemoryPool::find_suitable_block(size_t requested) {
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
  logger->info("Requesting, Used Pool size: {} Available Pool Size: {} Pool "
               "size: {}",
               this->used_pool.size(), this->available_pool.size(),
               memory->size);
}
