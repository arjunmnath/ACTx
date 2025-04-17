#pragma once

#include "memory.h"
#include "types.h"
#include <memory>
#include <unordered_map>
#include <vector>

typedef struct {
  Memory memory;
  uint32 size;
} MemoryItem;

class MemoryPool {
private:
  std::vector<MemoryItem> pool;

public:
  std::shared_ptr<Memory> requestMemoryFromPool();
};
