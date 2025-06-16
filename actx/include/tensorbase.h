#pragma once

#include "device_type.h"
#include "types.h"
class TensorBase {
private:
  /*Layout tensor_layout;*/
  DeviceType _device_type;

public:
  DType dtype;

  void view();
  void reshape();
};
