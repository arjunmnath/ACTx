#pragma once

#include "types.h"
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device_type.h"
#include "storage.h"
#include <mutex>
#include <string>

class Memory {
private:
  void *data_ptr;
  std::mutex _lock;
  DeviceType _type;

public:
  int size;
  DType dtype;
  std::unique_ptr<Storage> storage;
  Memory(DeviceType type, int size, DType dtype);
  bool operator<(const Memory &other) const { return size < other.size; }
  void acquire_lock();
  void release_lock();
  void guarded_lock();
};

class MemoryPimpl {
public:
  Memory *allocate(DeviceType type, int size, DType);
  void deallocate(void *ptr);
};
