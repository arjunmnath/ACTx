#include "device_type.h"
#include "dispather.h"
#include "op_types.h"
#include "tensor.h"

void Dispather::call(OPType op, DeviceType device, const Tensor &a,
                     const Tensor &b, Tensor &result) {
  Operation *operation = this->_register->get(op, device);
  operation->func(a, b, result);
}
