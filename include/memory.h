#pragma once

#include "types.h"
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device_type.h"
#include "storage.h"
#include <mutex>

class Memory {
private:
  std::mutex _lock;
  DeviceType _type;

public:
  void *data_ptr;
  size_t size;
  DType dtype;
  std::unique_ptr<Storage> storage;
  Memory(DeviceType type, int size, DType dtype);
  bool operator<(const Memory &other) const { return size < other.size; }
  bool does_live_on(DeviceType type);
  void acquire_lock();
  void release_lock();
  void guarded_lock();
};
