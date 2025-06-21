#pragma once

#include "device_type.h"
#include "storage.h"
#include "types.h"
#include <iostream>
#include <mutex>
#include <vector>

class Memory {
private:
  std::mutex _lock;

public:
  void *data_ptr;
  size_t size;
  DeviceType device;
  DType dtype;
  Storage *storage;
  Memory(DeviceType type, size_t count, DType dtype);
  static void copy(Memory *src, Memory *dest);
  static void copy_from_vector(std::vector<type_variant> src,
                               std::shared_ptr<Memory> dest);
  static void copy_to_vector(std::shared_ptr<Memory> src,
                             std::vector<type_variant> dest);
  bool does_live_on(DeviceType type);
  void acquire_lock();
  void release_lock();
  void guarded_lock();

  ~Memory() { std::cout << "Memory destroyed (size=" << size << ")\n"; }
};
