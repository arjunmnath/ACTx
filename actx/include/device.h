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
  virtual void sqrt(const Tensor *input, Tensor *output) = 0;
  virtual void exp(const Tensor *input, Tensor *output) = 0;
  virtual void log(const Tensor *input, Tensor *output) = 0;
  virtual void log10(const Tensor *input, Tensor *output) = 0;
  virtual void log2(const Tensor *input, Tensor *output) = 0;

  // trignometric
  virtual void sin(const Tensor *input, Tensor *output) = 0;
  virtual void cos(const Tensor *input, Tensor *output) = 0;
  virtual void tan(const Tensor *input, Tensor *output) = 0;
  virtual void asin(const Tensor *input, Tensor *output) = 0;
  virtual void acos(const Tensor *input, Tensor *output) = 0;
  virtual void atan(const Tensor *input, Tensor *output) = 0;
  virtual void atan2(const Tensor *x, const Tensor *y, Tensor *output) = 0;

  // hyperbolic
  virtual void sinh(const Tensor *input, Tensor *output) = 0;
  virtual void cosh(const Tensor *input, Tensor *output) = 0;
  virtual void tanh(const Tensor *input, Tensor *output) = 0;
  virtual void asinh(const Tensor *input, Tensor *output) = 0;
  virtual void acosh(const Tensor *input, Tensor *output) = 0;
  virtual void atanh(const Tensor *input, Tensor *output) = 0;

  // basic inits
  virtual void ones(Tensor *a) = 0;
  virtual void zeros(Tensor *a) = 0;
  virtual void eye(Tensor *a) = 0;
  virtual void full(Tensor *n, Tensor *result) = 0;

  // advanced inits;
  virtual void rand(Tensor *a, Tensor *meta) = 0;
};
