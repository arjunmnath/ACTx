#pragma once

#include "device_type.h"
#include "memory.h"
#include "tensor.h"
#include <string>

class Device {
private:
  std::string _name;

public:
  std::string name() { return this->_name; }
  virtual void add(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void sub(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void mul(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void div(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void matmul(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void pow(const Tensor *a, const Tensor *b, Tensor *result) = 0;

  // Comparison operators
  virtual void logical_e(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void logical_ne(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void logical_gt(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void logical_gte(const Tensor *a, const Tensor *b,
                           Tensor *result) = 0;
  virtual void logical_lt(const Tensor *a, const Tensor *b, Tensor *result) = 0;
  virtual void logical_lte(const Tensor *a, const Tensor *b,
                           Tensor *result) = 0;

  // Mathematical operations
  Tensor exp(bool inplace);
  Tensor log(bool inplace);

  // TODO: modify this to have a numpy like behaviour
  bool all();
  bool any();
  Tensor sqrt(bool inplace);
};
