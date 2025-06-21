#include "memory.h"
#include "main.h"
#include "storage.h"
#include "utility.h"
#include <cstring>
#include <stdexcept>

bool Memory::does_live_on(DeviceType type) { return this->device == type; }

void Memory::copy(Memory *src, Memory *dest) {
  id<MTLBuffer> buffer = src->storage->metal;
  id<MTLBuffer> bufferout = dest->storage->metal;

  assert(src->size <= dest->size);
  // NOTE: add more methods
  if (src->device == DeviceType::MPS && dest->device == DeviceType::MPS) {
    memcpy(dest->data_ptr, src->data_ptr, src->size);
  }
};
void Memory::copy_from_vector(std::vector<type_variant> src,
                              std::shared_ptr<Memory> dest) {}
void Memory::copy_to_vector(std::shared_ptr<Memory> src,
                            std::vector<type_variant> dest) {}
Memory::Memory(DeviceType type, size_t count, DType dtype) {
  this->device = type;
  this->size = count;
  this->dtype = dtype;
  switch (type) {
  case DeviceType::MPS: {
#ifdef __APPLE__
    this->storage = new Storage;
    mps->createEmptyBuffer(count, dtype, this->storage);
    this->data_ptr = [this->storage->metal contents];
#else
    throw std::runtime_error("Metal Not available");
#endif
    break;
  }
  case DeviceType::CPU:
    break;
  case DeviceType::WEBGPU:
    break;
  default:
    throw std::logic_error("unkown device type");
  }
}
