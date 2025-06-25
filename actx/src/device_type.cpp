#include "device_type.h"

std::string getDeviceName(DeviceType device) {
  switch (device) {
  case DeviceType::MPS:
    return "MPS";
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::WEBGPU:
    return "WEBGPU";
  default:
    return "unknown device";
  }
}
