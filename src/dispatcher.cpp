#include "dispatcher.h"
#include "device_type.h"
#include "main.h"
#include "op_types.h"
#include "tensor.h"
#include <iostream>
#include <optional>
#include <stdexcept>

void Dispatcher::call(
    OPType op, DeviceType device, Tensor &a,
    const std::optional<std::reference_wrapper<Tensor>> &b,
    const std::optional<std::reference_wrapper<Tensor>> &result) {
  Operation *operation = this->_register->get(op, device);
  if (operation == nullptr) {
    throw std::logic_error("operation not found");
  }
  operation->func(a, b, result);
}

void Dispatcher::init_register() {
  this->_register->register_op(
      OPType::ADD, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        mps->add(a, b->get(), result->get());
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
  this->_register->register_op(
      OPType::SUB, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        mps->sub(a, b->get(), result->get());
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
  this->_register->register_op(
      OPType::MUL, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        mps->mul(a, b->get(), result->get());
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
  this->_register->register_op(
      OPType::DIV, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        mps->div(a, b->get(), result->get());
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
  this->_register->register_op(
      OPType::ONES_INIT, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        assert(!b.has_value() && !result.has_value());
        mps->ones(a);
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
  this->_register->register_op(
      OPType::ZEROES_INIT, DeviceType::MPS,
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        assert(!b.has_value() && !result.has_value());
        mps->zeros(a);
      },
      [](Tensor &a, const std::optional<std::reference_wrapper<Tensor>> &b,
         const std::optional<std::reference_wrapper<Tensor>> &result) -> void {
        // TODO: do this
      });
}
