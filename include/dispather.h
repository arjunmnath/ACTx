#pragma once

#include "device_type.h"
#include "op_register.h"
#include "op_types.h"
#include "tensor.h"
#include <memory>
class Dispather {
private:
  std::unique_ptr<OpRegister> _register = std::make_unique<OpRegister>();

public:
  void call(OPType op, DeviceType device, const Tensor &a, const Tensor &b,
            Tensor &result);
  void init_register();
};
