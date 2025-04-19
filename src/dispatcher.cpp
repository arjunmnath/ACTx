#include "dispatcher.h"
#include "device_type.h"
#include "op_types.h"
#include "tensor.h"
#include <iostream>
#include <optional>

void Dispatcher::call(
    OPType op, DeviceType device, const Tensor &a,
    const std::optional<std::reference_wrapper<const Tensor>> &b,
    const std::optional<std::reference_wrapper<Tensor>> &result) {
  Operation *operation = this->_register->get(op, device);
  operation->func(a, b, result);
}

void Dispatcher::init_register() { std::cout << 'init' << std::endl; }
