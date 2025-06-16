#pragma once

#include "device_type.h"
#include "op_register.h"
#include "op_types.h"
#include "tensor.h"
#include <memory>
#include <optional>

class Dispatcher {
private:
  std::unique_ptr<OpRegister> _register = std::make_unique<OpRegister>();

public:
  void call(OPType op, DeviceType device, Tensor *a, Tensor *b = nullptr,
            Tensor *result = nullptr);
  void init_register();
};
