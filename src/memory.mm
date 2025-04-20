#include "memory.h"
#include "main.h"
#include "storage.h"
#include "utility.h"
#include <stdexcept>

bool Memory::does_live_on(DeviceType type) { return this->device == type; }

Memory::Memory(DeviceType type, size_t count, DType dtype) {
  this->device = type;
  this->size = count;
  this->dtype = dtype;
  switch (type) {
  case DeviceType::MPS:
#ifdef __APPLE__
    this->storage = std::make_unique<Storage>();
    this->storage->metal = mps->createEmptyBuffer(count, dtype);
    this->data_ptr = (void *)[this->storage->metal contents];
#else
    throw std::runtime_error("Metal Not available");
#endif // __APPLE__
    break;
  case DeviceType::CPU:
    break;
  case DeviceType::WEBGPU:
    break;
  default:
    throw std::logic_error("unkown device type");
  }
}
