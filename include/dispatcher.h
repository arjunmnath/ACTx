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
  void
  call(OPType op, DeviceType device, Tensor &a,
       const std::optional<std::reference_wrapper<Tensor>> &b = std::nullopt,
       const std::optional<std::reference_wrapper<Tensor>> &result =
           std::nullopt);
  void init_register();
};
