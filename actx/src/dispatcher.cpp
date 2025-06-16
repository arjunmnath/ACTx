#include "dispatcher.h"
#include "device_type.h"
#include "main.h"
#include "op_types.h"
#include "tensor.h"
#include <iostream>
#include <optional>
#include <stdexcept>

#define REGISTER_OP(OP, DEVICE, FUNC, BACKWARD)                                \
  this->_register->register_op(                                                \
      OPType::OP, DeviceType::DEVICE,                                          \
      [](Tensor * a, Tensor * b, Tensor * result) -> void FUNC,                \
      [](Tensor * a, Tensor * b, Tensor * result) -> void BACKWARD)

void Dispatcher::call(OPType op, DeviceType device, Tensor *a, Tensor *b,
                      Tensor *result) {
  Operation *operation = this->_register->get(op, device);
  if (operation == nullptr) {
    throw std::logic_error("operation not found");
  }
  operation->func(a, b, result);
}

void Dispatcher::init_register() {
  REGISTER_OP(ADD, MPS, { mps->add(a, b, result); }, {});
  REGISTER_OP(SUB, MPS, { mps->sub(a, b, result); }, {});
  REGISTER_OP(MUL, MPS, { mps->mul(a, b, result); }, {});
  REGISTER_OP(DIV, MPS, { mps->div(a, b, result); }, {});
  REGISTER_OP(LOGICAL_E, MPS, { mps->logical_e(a, b, result); }, {});
  REGISTER_OP(LOGICAL_NE, MPS, { mps->logical_ne(a, b, result); }, {});
  REGISTER_OP(LOGICAL_GT, MPS, { mps->logical_gt(a, b, result); }, {});
  REGISTER_OP(LOGICAL_GTE, MPS, { mps->logical_gte(a, b, result); }, {});
  REGISTER_OP(LOGICAL_LTE, MPS, { mps->logical_lte(a, b, result); }, {});
  REGISTER_OP(LOGICAL_LT, MPS, { mps->logical_lt(a, b, result); }, {});
  REGISTER_OP(ONES_INIT, MPS,
              {
                assert(b == nullptr && result == nullptr);
                mps->ones(a);
              },
              {});
  REGISTER_OP(ZEROES_INIT, MPS,
              {
                assert(b == nullptr && result == nullptr);
                mps->zeros(a);
              },
              {});
  REGISTER_OP(EYE_INIT, MPS,
              {
                assert(b == nullptr && result == nullptr);
                mps->eye(a);
              },
              {});
}
