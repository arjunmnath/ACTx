#include "tensor.h"
#include "mps.h"
#include "utility.cpp"
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
template <typename T> void Tensor<T>::_compte_stride() {
  /*strides[i] = (j=i+1 ∏ len(dims) - 1){shape[j]}*/
  int value = 1;
  this->stride.push_back(value);
  for (uint i = this->ndim - 1; i > 0; i--) {
    value *= this->dims[i];
    this->stride.insert(this->stride.begin(), value);
  }
}

template <typename T>
int Tensor<T>::_compute_offset(std::vector<int> indexes) const {
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
template <typename T>
int Tensor<T>::_compute_broadcast_index(
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

template <typename T>
std::vector<int>
Tensor<T>::_compute_broadcast_shape(const Tensor<T> *other) const {
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

template <typename T>
void Tensor<T>::throw_out_of_bound(std::vector<int> indexes) const {
  for (int i = 0; i < indexes.size(); i++) {
    if (indexes[i] >= this->dims[i]) {
      throw std::out_of_range("");
    }
  }
}
// ================================================================================================================================
//  KERNEL DISPATCHES
// ================================================================================================================================
template <typename T>
Tensor<T>
Tensor<T>::_dispatch_kernel_operation(const Tensor *other,
                                      std::string kernel_function) const {

  auto result_shape = this->_compute_broadcast_shape(other);

  id<MTLBuffer> result;
  result = device_mps->createEmptyBuffer<T>(std::accumulate(
      result_shape.begin(), result_shape.end(), 1, std::multiplies<int>()));
  id<MTLBuffer> lshape =
      device_mps->createBuffer(this->dims.data(), this->dims.size());
  id<MTLBuffer> rshape =
      device_mps->createBuffer(other->dims.data(), other->dims.size());
  id<MTLBuffer> target =
      device_mps->createBuffer(result_shape.data(), result_shape.size());

  std::vector<int> _ranks = {static_cast<int>(this->dims.size()),
                             static_cast<int>(other->dims.size()),
                             static_cast<int>(result_shape.size())};
  id<MTLBuffer> ranks = device_mps->createBuffer(_ranks.data(), _ranks.size());
  device_mps->execute_kernel_binary_with_broadcast(
      kernel_function, this->storage, other->storage, result, lshape, rshape,
      target, ranks);
  return Tensor(result, result_shape);
}

template <typename T>
Tensor<T>
Tensor<T>::_dispatch_kernel_operation_inplace(const Tensor *other,
                                              std::string kernel_function) {
  auto result_shape = this->_compute_broadcast_shape(other);
  id<MTLBuffer> lshape =
      device_mps->createBuffer(this->dims.data(), this->dims.size());
  id<MTLBuffer> rshape =
      device_mps->createBuffer(other->dims.data(), other->dims.size());
  id<MTLBuffer> target =
      device_mps->createBuffer(result_shape.data(), result_shape.size());

  std::vector<int> _ranks = {static_cast<int>(this->dims.size()),
                             static_cast<int>(other->dims.size()),
                             static_cast<int>(result_shape.size())};
  id<MTLBuffer> ranks = device_mps->createBuffer(_ranks.data(), _ranks.size());
  device_mps->execute_kernel_binary_with_broadcast(
      kernel_function, this->storage, other->storage, this->storage, lshape,
      rshape, target, ranks);
  this->dims = result_shape;
  return *this;
}
// ================================================================================================================================
// CONSTRUCTORS
// ================================================================================================================================
template <typename T>
Tensor<T>::Tensor(std::vector<int> dims, bool requires_grad) {
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  this->dims = dims;
  this->ndim = dims.size();
  this->storage = device_mps->createEmptyBuffer<T>(size);
  this->data_ptr = (T *)[this->storage contents];
  this->_compte_stride();
  this->requires_grad = requires_grad;
}

template <typename T>
Tensor<T>::Tensor(id<MTLBuffer> buffer, std::vector<int> dims,
                  bool requires_grad) {
  this->storage = buffer;
  this->data_ptr = (T *)[this->storage contents];
  this->dims = dims;
  this->ndim = dims.size();
  this->_compte_stride();
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
}

template <typename T>
Tensor<T>::Tensor(std::vector<T> &values, std::vector<int> dims,
                  bool requires_grad) {
  if (values.size() == 0) {
    throw std::runtime_error("values expected");
  }
  this->storage = device_mps->createBuffer(values.data(), values.size());
  this->data_ptr = (T *)[this->storage contents];
  this->dims = dims;
  this->ndim = dims.size();
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
template <typename T>
Tensor<T> Tensor<T>::ones(std::vector<int> shape, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);
  device_mps->execute_kernel_init("__ones__", result, meta);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(std::vector<int> shape, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);
  device_mps->execute_kernel_init("__zeros__", result, meta);
  return Tensor(result, shape);
}

template <typename T> Tensor<T> Tensor<T>::eye(int n, std::string dtype) {
  std::vector<int> shape = {n, n};
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(n * n);
  device_mps->execute_kernel_init("__eye__", result, meta);
  return Tensor(result, shape);
}
template <typename T>
Tensor<T> Tensor<T>::empty(std::vector<int> shape, std::string dtype) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);

  return Tensor(result, shape);
}
template <typename T>
Tensor<T> Tensor<T>::full(std::vector<int> shape, T n, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(shape[0] * shape[1]);

  std::vector<T> value = {n};
  id<MTLBuffer> seed = device_mps->createBuffer(value.data(), 1);
  device_mps->execute_kernel_unary("__full__", result, seed, meta);
  return Tensor(result, shape);
}
template <typename T>
Tensor<T> Tensor<T>::clone(Tensor<T> *other, std::string dtype) {
  id<MTLBuffer> newBuffer = device_mps->clone(other->storage);
  return Tensor(newBuffer, other->dims);
}

// TODO: configure the seed;
template <typename T>
Tensor<T> Tensor<T>::rand(std::vector<int> shape, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __rand<T>();
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::randn(std::vector<int> shape, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), 2);

  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randn<T>();
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::normal(std::vector<int> shape, float mean, float stddev,
                            std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randn<T>(mean, stddev);
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::randint(std::vector<int> shape, int min, int max,
                             std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());

  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __randint(min, max);
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::poission(Tensor &other, std::string dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(other.dims.data(), other.dims.size());
  int size = other.size;
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __poisson(other.data_ptr[i]);
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, other.dims);
}

template <typename T>
Tensor<T> Tensor<T>::bernoulli(Tensor &other, std::string dtype) {
  id<MTLBuffer> meta =
      device_mps->createBuffer(other.dims.data(), other.dims.size());
  int size = other.size;
  std::vector<T> data(size, 0);
  for (int i = 0; i < size; i++) {
    data[i] = __bernoulli(other.data_ptr[i]);
  }
  id<MTLBuffer> result = device_mps->createBuffer<T>(data.data(), size);
  return Tensor(result, other.dims);
}

// ================================================================================================================================
// GETTERS & SETTERS
// ================================================================================================================================
template <typename T> std::vector<int> Tensor<T>::strides() {
  return this->stride;
}

template <typename T>
template <typename... Args>
double Tensor<T>::getElement(Args... indexes) const {
  std::vector<int> indices = {indexes...};
  this->throw_out_of_bound(indices);
  int offset = this->_compute_offset(indices);
  return this->data_ptr[offset];
}

template <typename T>
template <typename... Args>
void Tensor<T>::setElement(T value, Args... indexes) {
  int indices[] = {indexes...};
  this->throw_out_of_bound(indices);
  int offset = this->_compute_offset(indices);
  this->data_ptr[offset] = value;
}
// ================================================================================================================================
// Arithemetic
// ================================================================================================================================
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__add__")
                 : this->_dispatch_kernel_operation(other, "__add__");
}
template <typename T>
Tensor<T> Tensor<T>::sub(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__sub__")
                 : this->_dispatch_kernel_operation(other, "__sub__");
}

template <typename T>
Tensor<T> Tensor<T>::mul(const Tensor *other, bool inplace) {
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__mul__")
                 : this->_dispatch_kernel_operation(other, "__mul__");
}

// TODO: fix this division by zero checking
template <typename T>
Tensor<T> Tensor<T>::div(const Tensor *other, bool inplace) {
  Tensor<T> zeros = Tensor<T>::zeros(other->dims);
  if (other->logical_e(&zeros).any()) {
    throw std::runtime_error("division by zero");
  }
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "__div__")
                 : this->_dispatch_kernel_operation(other, "__div__");
}
// TODO: multiple quick fixed has be done retest this method
template <typename T> Tensor<T> Tensor<T>::matmul(const Tensor *other) const {
  throw std::logic_error("not implemented");
  if (this->dims[1] != other->dims[0]) {
    throw std::runtime_error("shape contraint issue");
  }
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
  id<MTLBuffer> result;
  result = device_mps->createEmptyBuffer<T>(this->dims[0] * other->dims[1]);
  device_mps->execute_kernel_binary("__matmul__", this->storage, other->storage,
                                    other->storage, result);
  return Tensor(result, std::vector<int>{this->dims[0], other->dims[1]});
}

template <typename T> Tensor<T> Tensor<T>::pow(float exp, bool inplace) {
  std::vector<float> e = {exp};
  id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 3);
  id<MTLBuffer> exponent = device_mps->createBuffer(e.data(), 1);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer<T>(this->size);
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
template <typename T>
Tensor<T> Tensor<T>::logical_e(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_e");
}
template <typename T>
Tensor<T> Tensor<T>::logical_ne(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_ne");
}
template <typename T>
Tensor<T> Tensor<T>::logical_gt(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }

  return this->_dispatch_kernel_operation(other, "logical_gt");
}

template <typename T>
Tensor<T> Tensor<T>::logical_gte(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_gte");
}

template <typename T>
Tensor<T> Tensor<T>::logical_lt(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_lt");
}

template <typename T>
Tensor<T> Tensor<T>::logical_lte(const Tensor *other) const {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->_dispatch_kernel_operation(other, "logical_lte");
}

// Mathematical operations

template <typename T> Tensor<T> Tensor<T>::exp(bool inplace) {
  id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 2);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer<T>(this->size);
    device_mps->execute_kernel_unary("exp", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("exp", this->storage, this->storage, meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::log(bool inplace) {
  id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 2);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer<T>(this->size);
    device_mps->execute_kernel_unary("log", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("log", this->storage, this->storage, meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

template <typename T> bool Tensor<T>::all() {
  bool allTrue = true;

  for (int i = 0; i < this->size; i++) {
    if (false == this->data_ptr[i]) {
      allTrue = false;
    }
  }
  return allTrue;
}
template <typename T> bool Tensor<T>::any() {
  bool anyTrue = false;
  for (int i = 0; i < this->size; i++) {
    if (this->data_ptr[i]) {
      anyTrue = true;
    }
  }
  return anyTrue;
}

template <typename T> Tensor<T> Tensor<T>::sqrt(bool inplace) {
  id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 2);
  id<MTLBuffer> result;
  if (!inplace) {
    result = device_mps->createEmptyBuffer<T>(this->size);
    device_mps->execute_kernel_unary("sqrt", this->storage, result, meta);
  } else {
    device_mps->execute_kernel_unary("sqrt", this->storage, this->storage,
                                     meta);
  }
  return inplace ? *this : Tensor(result, this->dims);
}

template <typename T> Tensor<T> Tensor<T>::transpose() const {}

template <typename T> void Tensor<T>::print() const {
  T *ptr = (T *)[this->storage contents];
  std::cout << ptr[0] << std::endl;
}
template <typename T> void Tensor<T>::print_matrix() const {
  assert(this->stride.size() == 2);
  for (int i = 0; i < this->dims[0]; i++) {
    for (int j = 0; j < this->dims[1]; j++) {
      std::cout << this->data_ptr[this->stride[0] * i + this->stride[1] * j]
                << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
