#include "memory_pool.h"
#include "device_type.h"
#include "main.h"
#include "memory.h"
#include "types.h"
#include "utility.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

size_t MemoryPool::_compute_pool_size(size_t requested_count) {
  /*
   * requested_count is no of elements
   * requested_count(11): 1011 -> 4 (block_size: 16)
   * requested_count(32): 100000 -> 5 (block_size: 32)
   */
  return std::max(static_cast<int>(pow(2, ceil(log2(requested_count)))), 2);
}

Memory *MemoryPool::request_memory(DeviceType device, size_t length,
                                   DType dtype) {

  // std::cout << length << std::endl;
  int required_block_size = this->_compute_pool_size(length);
  int required_block_byte_size = required_block_size * getDTypeSize(dtype);
  Memory *suitable_block =
      this->find_suitable_block(device, dtype, required_block_byte_size);
  if (nullptr == suitable_block) {
    Memory *memory = new Memory(device, required_block_byte_size, dtype);
    this->used_pool.insert(memory);
    logger->info(COLOR("Requesting(allocating), ", BOLD_CYAN) +
                     COLOR("Used Pool size: {} ", BOLD_RED) +
                     COLOR("Available Pool Size: {} ", BOLD_GREEN) +
                     COLOR("Pool size: ", BOLD_CYAN) +
                     COLOR("{} bytes ", BOLD_WHITE) +
                     COLOR("Requested Size: ", BOLD_CYAN) +
                     COLOR("{} bytes", BOLD_WHITE),
                 this->used_pool.size(), this->available_pool.size(),
                 required_block_byte_size, length * getDTypeSize(dtype));
    return memory;
  }
  this->used_pool.insert(suitable_block);
  this->available_pool.erase(suitable_block);
  logger->info(
      COLOR("Requesting, ", BOLD_CYAN) +
          COLOR("Used Pool size: {} ", BOLD_RED) +
          COLOR("Available Pool Size: {} ", BOLD_GREEN) +
          COLOR("Pool size: ", BOLD_CYAN) + COLOR("{} bytes ", BOLD_WHITE) +
          COLOR("Requested Size: ", BOLD_CYAN) + COLOR("{} bytes", BOLD_WHITE),
      this->used_pool.size(), this->available_pool.size(),
      required_block_byte_size, length * getDTypeSize(dtype));

  return suitable_block;
}

Memory *MemoryPool::find_suitable_block(DeviceType device, DType dtype,
                                        size_t requested_size) {

  Memory *item = nullptr;
  for (auto it = this->available_pool.begin(); it != this->available_pool.end();
       ++it) {
    if ((*it)->bytesize >= requested_size &&
        (item == nullptr ? 1e10 : item->bytesize) >= (*it)->bytesize &&
        (*it)->dtype == dtype && (*it)->device == device) {
      item = *it;
    }
  }

  if (nullptr == item || item->bytesize >= requested_size * 2) {
    return nullptr;
  }
  return item;
}

void MemoryPool::return_memory(Memory *memory) {
  auto it = std::find_if(used_pool.begin(), used_pool.end(),
                         [&memory](const Memory *m) { return m == memory; });

  if (it != used_pool.end()) {
    this->used_pool.erase(it);
    this->available_pool.insert(memory);
  } else {
    logger->warn(
        COLOR("Tried to return memory that wasn't in used_pool!", BOLD_RED));
  }
  logger->info(
      COLOR("Returning, ", BOLD_YELLOW) +
          COLOR("Used Pool size: {} ", BOLD_RED) +
          COLOR("Available Pool Size: {} ", BOLD_GREEN) +
          COLOR("Pool size: ", BOLD_CYAN) + COLOR("{} bytes ", BOLD_WHITE),
      this->used_pool.size(), this->available_pool.size(), memory->bytesize);
}
