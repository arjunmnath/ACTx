#include "memory.h"
#include "mps.h"
#include <stdexcept>

Memory::Memory(DeviceType type, int size, DType dtype) {
  this->_type = type;
  this->size = size;
  this->dtype = dtype;
  switch (type) {
  case DeviceType::MPS:
#ifdef __APPLE__
    this->storage = new Storage;
    // TODO: fix this mps construction and use a global mps object
    this->storage->metal = MPS().createEmptyBuffer(size, dtype);
    this->data_ptr = (void *)[this->storage->metal contents];
#else
    throw std::runtime_error("Metal Not available");
#endif // __APPLE__
    break;
  case DeviceType::CPU:
    break;
  case DeviceType::WEBGPU:
    break;
  }
}
