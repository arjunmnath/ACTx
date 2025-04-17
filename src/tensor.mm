#include "tensor.h"
#include "mps.h"
#include "memory.h"
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
  int value = 1;
  this->stride.push_back(value);
  for (uint i = this->ndim - 1; i > 0; i--) {
    value *= this->dims[i];
    this->stride.insert(this->stride.begin(), value);
  }
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
int Tensor::_compute_broadcast_index(
    int flat_index, const std::vector<int> &source_shape,
    const std::vector<int> &target_shape) const {
  int source_rank = source_shape.size();
  int target_rank = target_shape.size();
  int source_index = 0;
  int stride = 1;

  for (int i = target_rank - 1; i >= 0; --i) {
    int target_dim = target_shape[i];
    int coord = (flat_index % target_dim);

    if (i >= target_rank - source_rank) {
      int source_dim = source_shape[i - (target_rank - source_rank)];
      if (source_dim > 1) {
        source_index += coord * stride;
      }
    }

    flat_index /= target_dim;
    if (i >= target_rank - source_rank) {
      stride *= (source_shape[i - (target_rank - source_rank)] > 1
                     ? source_shape[i - (target_rank - source_rank)]
                     : 1);
    }
  }

  return source_index;
}

std::vector<int> Tensor::_compute_broadcast_shape(const Tensor *other) const {
  int max_rank = std::max(other->dims.size(), this->dims.size());

  std::vector<int> rev_shape1 = this->dims;
  std::vector<int> rev_shape2 = other->dims;

  std::reverse(rev_shape1.begin(), rev_shape1.end());
  std::reverse(rev_shape2.begin(), rev_shape2.end());

  rev_shape1.resize(max_rank, 1);
  rev_shape2.resize(max_rank, 1);

  std::vector<int> result(max_rank);

  int dim1, dim2;
  for (int i = 0; i < max_rank; i++) {
    dim1 = rev_shape1[i];
    dim2 = rev_shape2[i];
    if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
      result[i] = std::max(dim1, dim2);
    } else {
      throw std::invalid_argument("Shapes not broadcastable");
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
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
//  KERNEL DISPATCHES
// ================================================================================================================================
Tensor Tensor::_dispatch_kernel_operation(const Tensor *other,
                                          std::string kernel_function) const {

  auto result_shape = this->_compute_broadcast_shape(other);

  id<MTLBuffer> result;
  // TODO: fix needed
  result = device_mps->createEmptyBuffer(
      std::accumulate(result_shape.begin(), result_shape.end(), 1,
                      std::multiplies<int>()),
      this->dtype);
  id<MTLBuffer> lshape = device_mps->createBuffer(
      (void *)this->dims.data(), this->dims.size(), this->dtype);
  id<MTLBuffer> rshape = device_mps->createBuffer(
      (void *)other->dims.data(), other->dims.size(), this->dtype);
  id<MTLBuffer> target = device_mps->createBuffer(
      result_shape.data(), result_shape.size(), this->dtype);

  std::vector<int> _ranks = {static_cast<int>(this->dims.size()),
                             static_cast<int>(other->dims.size()),
                             static_cast<int>(result_shape.size())};
  id<MTLBuffer> ranks =
      device_mps->createBuffer(_ranks.data(), _ranks.size(), this->dtype);
  device_mps->execute_kernel_binary_with_broadcast(
      kernel_function, this->storage, other->storage, result, lshape, rshape,
      target, ranks);
  return Tensor(result, result_shape);
}

Tensor Tensor::_dispatch_kernel_operation_inplace(const Tensor *other,
                                                  std::string kernel_function) {
  auto result_shape = this->_compute_broadcast_shape(other);
  id<MTLBuffer> lshape = device_mps->createBuffer(
      this->dims.data(), this->dims.size(), this->dtype);
  id<MTLBuffer> rshape = device_mps->createBuffer(
      (void *)other->dims.data(), other->dims.size(), this->dtype);
  id<MTLBuffer> target = device_mps->createBuffer(
      result_shape.data(), result_shape.size(), this->dtype);

  std::vector<int> _ranks = {static_cast<int>(this->dims.size()),
                             static_cast<int>(other->dims.size()),
                             static_cast<int>(result_shape.size())};
  id<MTLBuffer> ranks =
      device_mps->createBuffer(_ranks.data(), _ranks.size(), this->dtype);
  device_mps->execute_kernel_binary_with_broadcast(
      kernel_function, this->storage, other->storage, this->storage, lshape,
      rshape, target, ranks);
  this->dims = result_shape;
  return *this;
}
// ================================================================================================================================
// CONSTRUCTORS
// ================================================================================================================================
Tensor::Tensor(std::vector<int> dims, bool requires_grad) {
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  this->dims = dims;
  this->ndim = dims.size();
  this->dtype = DType::float32;
  this->storage = device_mps->createEmptyBuffer(size, this->dtype);
  this->_compte_stride();
  this->data_ptr = (float *)[this->storage contents];
  this->requires_grad = requires_grad;
}

Tensor::Tensor(id<MTLBuffer> buffer, std::vector<int> dims,
               bool requires_grad) {
  this->storage = buffer;
  this->dims = dims;
  this->dtype = DType::float32;
  this->ndim = dims.size();
  this->_compte_stride();
  this->data_ptr = (float *)[this->storage contents];
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
}

// TODO: fix the vector<float> and dtype mismatch
Tensor::Tensor(std::vector<float> &values, std::vector<int> dims,
               bool requires_grad) {
  if (values.size() == 0) {
    throw std::runtime_error("values expected");
  }
  this->dtype = DType::float32;
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
//                            INIT
// ================================================================================================================================
// 1) Ones & zeros: ✅
// 2) Empty:✅
// 3) Eye: ✅
// 4) Normal, bernoulli, poisson: ✅
// 5) Rand, randn, randint: ✅
// 6) Clone, tensor: ❌
// 7) Linspace, logspace, arange: ❌
// ================================================================================================================================
Tensor Tensor::ones(std::vector<int> shape, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer(size, dtype);
  device_mps->execute_kernel_init("__ones__", result, meta);
  return Tensor(result, shape);
}

Tensor Tensor::zeros(std::vector<int> shape, DType dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(shape.data(), shape.size(), dtype);
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer(size, dtype);
  device_mps->execute_kernel_init("__zeros__", result, meta);
  return Tensor(result, shape);
}

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
// ================================================================================================================================
// Arithemetic
// ================================================================================================================================
Tensor Tensor::add(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__add__")
                 : this->_dispatch_kernel_operation(other, "__add__");
}
Tensor Tensor::sub(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__sub__")
                 : this->_dispatch_kernel_operation(other, "__sub__");
}

Tensor Tensor::mul(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__mul__")
                 : this->_dispatch_kernel_operation(other, "__mul__");
}

// TODO: fix this division by zero checking
Tensor Tensor::div(const Tensor *other, bool inplace) {
  Tensor zeros = Tensor::zeros(other->dims);
  if (other->logical_e(&zeros).any()) {
    throw std::runtime_error("division by zero");
  }
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__div__")
                 : this->_dispatch_kernel_operation(other, "__div__");
}
// TODO: multiple quick fixed has be done retest this method
Tensor Tensor::matmul(const Tensor *other) const {
  throw std::logic_error("not implemented");
  if (this->dims[1] != other->dims[0]) {
    throw std::runtime_error("shape contraint issue");
  }
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3, this->dtype);
  id<MTLBuffer> result;
  result = device_mps->createEmptyBuffer(this->dims[0] * other->dims[1],
                                         this->dtype);
  device_mps->execute_kernel_binary("__matmul__", this->storage, other->storage,
                                    other->storage, result);
  return Tensor(result, std::vector<int>{this->dims[0], other->dims[1]});
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

// TODO: impelement this
Tensor Tensor::transpose() const { return Tensor::eye(5); }

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
