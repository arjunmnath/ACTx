#include "memory.h"
#include "mps.h"

Memory::Memory(DeviceType type, int size, DType dtype) {
  this->_type = type;
  switch (type) {
  case DeviceType::MPS:
    this->memory = new Storage;
    this->memory->metal = MPS().createEmptyBuffer(size, dtype);
    this->data_ptr = (void *)[this->memory->metal contents];
    // TODO: memory assignment logic pending
    break;
  case DeviceType::CPU:
    break;
  case DeviceType::WEBGPU:
    break;
  }
}
