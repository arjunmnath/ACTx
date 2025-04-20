#pragma once

#include "device_type.h"
#include "storage.h"
#include "types.h"
#include <mutex>

class Memory {
private:
  std::mutex _lock;

public:
  void *data_ptr;
  size_t size;
  DeviceType device;
  DType dtype;
  std::unique_ptr<Storage> storage;
  Memory(DeviceType type, size_t count, DType dtype);
  bool does_live_on(DeviceType type);
  void acquire_lock();
  void release_lock();
  void guarded_lock();
};
