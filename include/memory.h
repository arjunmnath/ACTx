#pragma once

#include "types.h"
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device_type.h"
#include <Metal/Metal.h>
#include <mutex>
#include <string>

union Storage {
  id<MTLBuffer> metal;
  void *cpu;
  Storage() {}
  ~Storage() {}
};

class Memory {
private:
  Storage *memory;
  void *data_ptr;
  std::mutex _lock;
  DeviceType _type;

public:
  int size;
  DType dtype;
  Memory(DeviceType type, int size, DType dtype = DType::float32);

  bool operator<(const Memory &other) const { return size < other.size; }
  void acquire_lock();
  void release_lock();
  void guarded_lock();
};
