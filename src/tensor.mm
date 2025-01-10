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
void Tensor<T>::throw_out_of_bound(std::vector<int> indexes) const {
  for (int i = 0; i < indexes.size(); i++) {
    if (indexes[i] >= this->dims[i]) {
      throw std::out_of_range("");
    }
  }
}

template <typename T>
Tensor<T>
Tensor<T>::_dispatch_kernel_operation(const Tensor *other,
                                      std::string kernel_function) const {
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
  id<MTLBuffer> result;
  result = device_mps->createEmptyBuffer<T>(this->size);
  device_mps->execute_kernel_binary(kernel_function, this->storage,
                                    other->storage, result, meta);
  return Tensor(result, this->dims);
}
template <typename T>
Tensor<T>
Tensor<T>::_dispatch_kernel_operation_inplace(const Tensor *other,
                                              std::string kernel_function) {
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
  device_mps->execute_kernel_binary(kernel_function, this->storage,
                                    other->storage, this->storage, meta);
  return *this;
}

template <typename T> Tensor<T>::Tensor(std::vector<int> dims) {
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  this->dims = dims;
  this->ndim = dims.size();
  this->storage = device_mps->createEmptyBuffer<T>(size);
  this->data_ptr = (T *)[this->storage contents];
  this->_compte_stride();
}

template <typename T>
Tensor<T>::Tensor(id<MTLBuffer> buffer, std::vector<int> dims) {
  this->storage = buffer;
  this->data_ptr = (T *)[this->storage contents];
  this->dims = dims;
  this->ndim = dims.size();
  this->_compte_stride();
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
}

template <typename T>
Tensor<T>::Tensor(std::vector<T> &values, std::vector<int> dims) {
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
}

// =====================================================================
//                            INIT
// =====================================================================
// 1) Ones & zeros: ✅
// 2) Empty:✅
// 3) Eye: ✅
// 4) Normal, bernoulli, poisson: ✅
// 5) Rand, randn, randint: ✅
// 6) Clone, tensor: ❌
// 7) Linspace, logspace, arange: ❌
// =====================================================================

template <typename T>
Tensor<T> Tensor<T>::ones(std::vector<int> shape, std::string dtype) {
  if (shape.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);
  device_mps->execute_kernel_init("init_ones", result, meta);
  return Tensor(result, shape);
}

template <typename T>
Tensor<T> Tensor<T>::zeros(std::vector<int> shape, std::string dtype) {
  if (shape.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);
  device_mps->execute_kernel_init("init_with_zeros", result, meta);
  return Tensor(result, shape);
}

template <typename T> Tensor<T> Tensor<T>::eye(int n, std::string dtype) {
  std::vector<int> shape = {n, n};
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(n * n);
  device_mps->execute_kernel_init("init_identity", result, meta);
  return Tensor(result, shape);
}
template <typename T>
Tensor<T> Tensor<T>::empty(std::vector<int> shape, std::string dtype) {
  if (shape.size() != 2) {
    throw std::runtime_error("shape contraint failed");
  }
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(size);

  return Tensor(result, shape);
}
template <typename T>
Tensor<T> Tensor<T>::full(std::vector<int> shape, int n, std::string dtype) {
  id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size());
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  id<MTLBuffer> result = device_mps->createEmptyBuffer<T>(shape[0] * shape[1]);

  std::vector<int> seed_vec = {n};
  id<MTLBuffer> seed = device_mps->createBuffer(seed_vec.data(), 1);
  device_mps->execute_kernel_unary("init_full", result, seed, meta);
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

// arithemetic Operators
//
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor *other, bool inplace) {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint issue");
  }
  return inplace ? this->_dispatch_kernel_operation_inplace(other, "add_matrix")
                 : this->_dispatch_kernel_operation(other, "add_matrix");
}
template <typename T>
Tensor<T> Tensor<T>::sub(const Tensor *other, bool inplace) {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint issue");
  }
  return inplace ? this->_dispatch_kernel_operation_inplace(other,
                                                            "subtract_matrix")
                 : this->_dispatch_kernel_operation(other, "subtract_matrix");
}

template <typename T>
Tensor<T> Tensor<T>::mul(const Tensor *other, bool inplace) {
  if (this->dims != other->dims || this->dims.size() != 2) {
    throw std::runtime_error("shape contraint issue");
  }
  return inplace ? this->_dispatch_kernel_operation_inplace(
                       other, "elementwise_multiply_matrix")
                 : this->_dispatch_kernel_operation(
                       other, "elementwise_multiply_matrix");
}

// TODO: fix this division by zero checking
template <typename T>
Tensor<T> Tensor<T>::div(const Tensor *other, bool inplace) {
  Tensor<T> zeros = Tensor<T>::zeros(other->dims);
  if (other->logical_e(&zeros).any() || this->dims != other->dims ||
      this->dims.size() != 2) {
    throw std::runtime_error("shape contraint issue");
  }

  return inplace ? this->_dispatch_kernel_operation_inplace(
                       other, "elementwise_divide_matrix")
                 : this->_dispatch_kernel_operation(
                       other, "elementwise_divide_matrix");
}

template <typename T> Tensor<T> Tensor<T>::matmul(const Tensor *other) const {
  if (this->dims[1] != other->dims[0]) {
    throw std::runtime_error("shape contraint issue");
  }
  std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
  id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
  id<MTLBuffer> result;
  result = device_mps->createEmptyBuffer<T>(this->dims[0] * other->dims[1]);
  device_mps->execute_kernel_binary("matrix_multiply", this->storage,
                                    other->storage, result, meta);
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
