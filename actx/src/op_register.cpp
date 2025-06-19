#include "op_register.h"
#include <functional>

void OpRegister::register_op(OPType op, DeviceType device, TensorOperation func,
                             std::function<void(OpNode *node)> backward) {
  Operation *operation = new Operation;
  operation->func = func;
  operation->backward = backward;

  this->ops[device].emplace(op, operation);
}

Operation *OpRegister::get(OPType op, DeviceType device) {
  return this->ops[device][op];
}
