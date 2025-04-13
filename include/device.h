#pragma once

#include "device_type.h"
#include "memory.h"
#include "tensor.h"
#include <string>

class Device {
private:
  std::string _name;

public:
  Memory<DeviceType> alloc();
  void sync();
  std::string name() { return this->_name; }

  Tensor add(const Tensor *other, bool inplace);

  Tensor sub(const Tensor *other, bool inplace);
  Tensor mul(const Tensor *other, bool inplace);
  Tensor div(const Tensor *other, bool inplace);
  Tensor matmul(const Tensor *other) const;
  Tensor pow(float exp, bool inplace);

  // Comparison operators
  Tensor logical_e(const Tensor *other) const;
  Tensor logical_ne(const Tensor *other) const;
  Tensor logical_gt(const Tensor *other) const;
  Tensor logical_gte(const Tensor *other) const;
  Tensor logical_lt(const Tensor *other) const;
  Tensor logical_lte(const Tensor *other) const;

  // Mathematical operations
  Tensor exp(bool inplace);
  Tensor log(bool inplace);

  // TODO: modify this to have a numpy like behaviour
  bool all();
  bool any();
  Tensor sqrt(bool inplace);
};
