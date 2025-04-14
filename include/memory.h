#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device_type.h"
#include <Metal/Metal.h>
#include <mutex>
#include <string>
/*#include <webgpu/webgpu.h>*/

union Storage {
  id<MTLBuffer> metal;
  void *cpu;
  Storage() {}
  ~Storage() {}
};
class Memory {
private:
  Storage *memory;
  std::mutex _lock;
  DeviceType _type;
  Memory(DeviceType type) {
    this->_type = type;
    switch (type) {
    case DeviceType::MPS:
      memory = new Storage;
      memory->metal = ;
      break;
    case DeviceType::CPU:
      break;
    case DeviceType::WEBGPU:
      break;
    }
  }

public:
  void acquire_lock();
  void release_lock();
  void guarded_lock();
};
