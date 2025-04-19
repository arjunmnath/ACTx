#include "tensor.h"
#include "main.h"
#include "memory.h"
#include "mps.h"
#include "types.h"
#include "utility.h"
#include <MacTypes.h>
#include <Metal/Metal.h>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

MPS *device_mps = new MPS();
// TODO: use better error time;
// ================================================================================================================================
// COMPUTES
// ================================================================================================================================
void Tensor::_compte_stride() {
  /*strides[i] = (j=i+1 ∏ len(dims) - 1){shape[j]}*/
  if (this->ndim == 0 || this->dims.empty()) {
    throw std::runtime_error("dims and ndim not initialized properly.");
  }

  assert(this->dims.size() == this->ndim &&
         "Mismatch between 'ndim' and 'dims' size");
  int value = 1;
  this->stride.clear();
  this->stride.push_back(value);
  assert(this->dims.size() == this->ndim);

  for (uint i = this->ndim - 1; i > 0; i--) {
    value *= this->dims[i];
    this->stride.push_back(value);
  }
  std::reverse(this->stride.begin(), this->stride.end());
}

int Tensor::_compute_offset(std::vector<int> indexes) const {
  int n = indexes.size();
  int offset = 0;
  if (n != this->stride.size()) {
    throw std::runtime_error("indexes size mismatch");
  }

  for (int i = 0; i < n; i++) {
    offset += indexes[i] * this->stride[i];
  }
  return offset;
}
// ================================================================================================================================

void Tensor::throw_out_of_bound(std::vector<int> indexes) const {
  for (int i = 0; i < indexes.size(); i++) {
    if (indexes[i] >= this->dims[i]) {
      throw std::out_of_range("");
    }
  }
}
// ================================================================================================================================
// CONSTRUCTORS
// ================================================================================================================================
Tensor::Tensor(std::vector<int> dims, DType dtype, bool requires_grad) {
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  this->dims = dims;
  this->ndim = dims.size();
  // TODO: change this to cpu
  this->device = DeviceType::MPS;
  this->dtype = dtype;
  this->memory = pool->request_memory(this->device, this->size, this->dtype);
  this->_compte_stride();
  this->data_ptr = (float *)[this->storage contents];
  this->requires_grad = requires_grad;
}

Tensor::Tensor(std::shared_ptr<Memory> memory, std::vector<int> dims,
               DType dtype, bool requires_grad) {
  this->dims = dims;
  this->memory = memory;
  this->dtype = dtype;
  // TODO: change this to cpu
  this->device = DeviceType::MPS;
  this->ndim = dims.size();
  this->_compte_stride();
  this->data_ptr = (float *)[this->storage contents];
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
}

// TODO: fix the vector<float> and dtype mismatch
Tensor::Tensor(std::vector<float> &values, std::vector<int> dims, DType dtpe,
               bool requires_grad) {
  if (values.size() == 0) {
    throw std::runtime_error("values expected");
  }
  this->dtype = dtype;
  this->storage =
      device_mps->createBuffer(values.data(), values.size(), this->dtype);
  this->dims = dims;
  this->ndim = dims.size();
  this->data_ptr = (float *)[this->storage contents];
  this->_compte_stride();
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
}

// ================================================================================================================================
// GETTERS & SETTERS
// ================================================================================================================================

std::vector<int> Tensor::strides() { return this->stride; }

template <typename... Args> double Tensor::getElement(Args... indexes) const {
  std::vector<int> indices = {indexes...};
  this->throw_out_of_bound(indices);
  int offset = this->_compute_offset(indices);
  return this->data_ptr[offset];
}

template <typename... Args>
void Tensor::setElement(float value, Args... indexes) {
  int indices[] = {indexes...};
  this->throw_out_of_bound(indices);
  int offset = this->_compute_offset(indices);
  this->data_ptr[offset] = value;
}

// TODO: impelement this
Tensor Tensor::transpose() const { throw std::logic_error("not implemented"); }

void Tensor::print() const {
  float *ptr = (float *)[this->storage contents];
  std::cout << ptr[0] << std::endl;
}
void Tensor::print_matrix() const {
  for (int i = 0; i < this->dims[0]; i++) {
    for (int j = 0; j < this->dims[1]; j++) {
      std::cout << this->data_ptr[this->stride[0] * i + this->stride[1] * j]
                << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

Tensor Tensor::execute_broadcastable_operation(OPType op, Tensor *other,
                                               bool inplace) {
  if (this->requires_grad || other->requires_grad) {
    this->requires_grad = other->requires_grad = true;
  }
  if (!inplace) {
    auto result_shape = compute_broadcast_shape(*this, *other);
    std::shared_ptr<Memory> result_memory = pool->request_memory(
        this->device,
        std::accumulate(result_shape.begin(), result_shape.end(), 1,
                        std::multiplies<int>()),
        this->dtype);
    return Tensor(result_memory, result_shape, this->dtype,
                  this->requires_grad);
  }
  dispatcher->call(op, this->device, *this, *other, *this);
  return *this;
}

Tensor Tensor::execute_init_operation(OPType op, std::vector<int> shape,
                                      DType dtype, bool requires_grad,
                                      DeviceType device) {
  std::shared_ptr<Memory> result_memory = pool->request_memory(
      device,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()),
      dtype);
  Tensor result(result_memory, shape, dtype, requires_grad);
  dispatcher->call(op, device, result, std::nullopt, std::nullopt);
  return result;
}

// ================================================================================================================================
// Arithemetic
// ================================================================================================================================
Tensor Tensor::add(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::ADD, other, inplace);
}
Tensor Tensor::sub(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::SUB, other, inplace);
}

Tensor Tensor::mul(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::MUL, other, inplace);
}

Tensor Tensor::div(Tensor *other, bool inplace) {
  // TODO: fix this division by zero checking
  /*
    Tensor zeros = Tensor::zeros(other->dims);
    if (other->logical_e(&zeros).any()) {
      throw std::runtime_error("division by zero");
    }
  */
  return execute_broadcastable_operation(OPType::DIV, other, inplace);
}

/*
Tensor Tensor::matmul(Tensor *other) const {
  // TODO: implement broadcastable matmul;
  throw std::logic_error("not implemented");
  if (this->dims[1] != other->dims[0]) {
    throw std::runtime_error("shape contraint issue");
  }
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  return Tensor(m, true);
}

Tensor Tensor::pow(float exp, bool inplace) {
  std::vector<float> e = {exp};
  id<MTLBuffer> meta =
      device_mps->createBuffer(this->dims.data(), 3, this->dtype);
  id<MTLBuffer> exponent = device_mps->createBuffer(e.data(), 1, this->dtype);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer(this->size, this->dtype);
    device_mps->execute_kernel_binary("elementwise_pow", this->storage,
                                      exponent, result, meta);
  } else {

    device_mps->execute_kernel_binary("elementwise_pow", this->storage,
                                      exponent, this->storage, meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

// Comparison operators
//
Tensor Tensor::logical_e(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_e");
}
Tensor Tensor::logical_ne(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_ne");
}
Tensor Tensor::logical_gt(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_gt");
}

Tensor Tensor::logical_gte(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_gte");
}

Tensor Tensor::logical_lt(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_lt");
}

Tensor Tensor::logical_lte(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_lte");
}

// Mathematical operations

Tensor Tensor::exp(bool inplace) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(this->dims.data(), 2, this->dtype);
  id<MTLBuffer> result;
  if (!inplace) {
    // TODO: fix fixed float
    result = device_mps->createEmptyBuffer(this->size, this->dtype);
    device_mps->execute_kernel_unary("exp", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("exp", this->storage, this->storage, meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

Tensor Tensor::log(bool inplace) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(this->dims.data(), 2, this->dtype);
  id<MTLBuffer> result;
  if (!inplace) {
    // TODO: fix fixed float
    result = device_mps->createEmptyBuffer(this->size, this->dtype);
    device_mps->execute_kernel_unary("log", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("log", this->storage, this->storage, meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

bool Tensor::all() {
  bool allTrue = true;

  for (int i = 0; i < this->size; i++) {
    if (false == this->data_ptr[i]) {
      allTrue = false;
    }
  }
  return allTrue;
}
bool Tensor::any() {
  bool anyTrue = false;
  for (int i = 0; i < this->size; i++) {
    if (this->data_ptr[i]) {
      anyTrue = true;
    }
  }
  return anyTrue;
}

Tensor Tensor::sqrt(bool inplace) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(this->dims.data(), 2, this->dtype);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer(this->size, this->dtype);
    device_mps->execute_kernel_unary("sqrt", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("sqrt", this->storage, this->storage,
                                     meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

*/

// ================================================================================================================================
//                            INIT
// ================================================================================================================================
// 1) Ones & zeros: ✅
// 2) Empty:✅
// 3) Eye: ✅
// 4) Normal, bernoulli, poisson: ✅
// 5) Rand, randn, randint: ✅
// 6) Clone, tensor: ❌
// 7) Linspace, logspace, arange: ❌
// =====================================================================================================================
Tensor Tensor::ones(std::vector<int> shape, DType dtype) {
  return Tensor::execute_init_operation(OPType::ONES_INIT, shape, dtype);
}

Tensor Tensor::zeros(std::vector<int> shape, DType dtype) {
  return Tensor::execute_init_operation(OPType::ZEROES_INIT, shape, dtype);
}

/*
Tensor Tensor::eye(int n, DType dtype) {
  std::vector<int> shape = {n, n};
 id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  id<MTLBuffer> result = device_mps->createEmptyBuffer(n * n, dtype);
  device_mps->execute_kernel_init("__eye__", result, meta);
  return Tensor(result, shape);
}

Tensor Tensor::empty(std::vector<int> shape, DType dtype) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer(size, dtype);

  return Tensor(result, shape);
}

// FIX: mismatch of type of n and dtype
template <typename T>
Tensor Tensor::full(std::vector<int> shape, T n, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result =
      device_mps->createEmptyBuffer(shape[0] * shape[1], dtype);

  std::vector<T> value = {n};
  id<MTLBuffer> seed = device_mps->createBuffer(value.data(), 1);
  device_mps->execute_kernel_unary("__full__", result, seed, meta);
  return Tensor(result, shape);
}
Tensor Tensor::clone(Tensor *other) {
  id<MTLBuffer> newBuffer = device_mps->clone(other->storage);
  return Tensor(newBuffer, other->dims);
}

// TODO: configure the seed && change vector type from float to dynamic;
Tensor Tensor::rand(std::vector<int> shape, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __rand();
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, shape);
}

Tensor Tensor::randn(std::vector<int> shape, DType dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), 2, dtype);

  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randn();
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, shape);
}

Tensor Tensor::normal(std::vector<int> shape, float mean, float stddev,
                      DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randn(mean, stddev);
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, shape);
}

Tensor Tensor::randint(std::vector<int> shape, int min, int max, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);

  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randint(min, max);
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, shape);
}
Tensor Tensor::poission(Tensor &other, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(other.dims.data(), other.dims.size(), dtype);
  int size = other.size;
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    // TODO: fix this concrete tempalte type
    data[i] = __poisson(other.data_ptr[i]);
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, other.dims);
}
Tensor Tensor::bernoulli(Tensor &other, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(other.dims.data(), other.dims.size(), dtype);
  int size = other.size;
  std::vector<float> data(size, 0);
  for (int i = 0; i < size; i++) {
    // TODO: fix this concrete tempalte type
    data[i] = __bernoulli(other.data_ptr[i]);
  }
  id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
  return Tensor(result, other.dims);
}
*/
