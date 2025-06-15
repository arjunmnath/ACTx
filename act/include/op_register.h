#pragma once

#include "device_type.h"
#include "op_types.h"
#include "tensor.h"
#include <functional>

using TensorOperation = std::function<void(Tensor *, Tensor *, Tensor *)>;

struct Operation {
  TensorOperation func;
  TensorOperation backward;
};

class OpRegister {
private:
  std::unordered_map<DeviceType, std::unordered_map<OPType, Operation *>> ops;

public:
  void register_op(OPType op, DeviceType device, TensorOperation func,
                   TensorOperation backward);
  Operation *get(OPType op, DeviceType device);
};
