#include "tensor.h"
#include "main.h"
#include "types.h"
#include "utility.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

// ================================================================================================================================
// COMPUTES
// ================================================================================================================================

Tensor *Tensor::view(std::vector<Slice> &slices) const {
  assert(slices.size() <= this->ndim);

  std::vector<int> view_dims = {1, 1};
  Tensor *view_tensor = new Tensor(this->memory, view_dims);

  view_tensor->ndim = static_cast<int>(slices.size());
  view_tensor->dims.resize(view_tensor->ndim);
  view_tensor->stride.resize(view_tensor->ndim);

  view_tensor->dtype = this->dtype;
  view_tensor->requires_grad = this->requires_grad;
  view_tensor->device = this->device;
  view_tensor->is_view = true;

  int new_offset_elements = this->offset_elements;
  for (int d = 0; d < view_tensor->ndim; d++) {
    int start = slices[d].start;
    int stop = slices[d].stop;
    int step = slices[d].step;

    if (start < 0)
      start += this->dims[d];
    if (stop < 0)
      stop += this->dims[d];
    if (start < 0)
      start = 0;
    if (stop > this->dims[d])
      stop = this->dims[d];
    if (stop < start)
      stop = start;

    int len = (stop - start + step - 1) / step;

    view_tensor->dims[d] = len;
    view_tensor->stride[d] = this->stride[d] * step;

    new_offset_elements += start * this->stride[d];
  }
  view_tensor->offset_elements = new_offset_elements;
  return view_tensor;
}
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

void Tensor::reinterpret_pointer(void *ptr) {
  switch (this->dtype) {
  case DType::int8:
    break;
  case DType::float16:
  case DType::int16:
    this->data_ptr = ptr;
    break;

  case DType::float32:
    this->data_ptr = (float *)ptr;
    break;

  case DType::int32:
    this->data_ptr = (int *)ptr;
    break;
  case DType::int64:
    this->data_ptr = ptr;
    break;
  default:
    throw std::invalid_argument("not implemented");
    break;
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
  this->offset_elements = 0;
  this->reinterpret_pointer(this->memory->data_ptr);
  this->_compte_stride();
  this->requires_grad = requires_grad;
}

Tensor::Tensor(std::shared_ptr<Memory> memory, std::vector<int> dims,
               DType dtype, bool requires_grad) {
  this->dims = dims;
  this->memory = memory;
  this->dtype = dtype;
  this->reinterpret_pointer(this->memory->data_ptr);
  // TODO: change this to cpu
  this->device = DeviceType::MPS;
  this->ndim = dims.size();

  this->offset_elements = 0;
  this->_compte_stride();
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
}

// FIX: fix the vector<float> and dtype mismatch and allocate memory and do
// memcpy
Tensor::Tensor(std::vector<float> &values, std::vector<int> dims, DType dtype,
               bool requires_grad) {
  if (values.size() == 0) {
    throw std::runtime_error("values expected");
  }
  this->dtype = dtype;
  this->dims = dims;
  this->ndim = dims.size();
  this->_compte_stride();
  this->offset_elements = 0;
  this->device = DeviceType::MPS;
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  assert(values.size() == this->size);
  this->memory = pool->request_memory(this->device, this->size, this->dtype);
  mps->copy_vector_to_buffer(values.data(), *this->memory,
                             values.size() * getDTypeSize(dtype));
  this->reinterpret_pointer(this->memory->data_ptr);
  this->requires_grad = requires_grad;
}

// ================================================================================================================================
// GETTERS & SETTERS
// ================================================================================================================================

std::vector<int> Tensor::strides() { return this->stride; }

float Tensor::_get_element(int offset) const {
  int total_offset = (offset + offset_elements);
  if (std::holds_alternative<int *>(this->data_ptr)) {
    return std::get<int *>(this->data_ptr)[total_offset];
  } else if (std::holds_alternative<float *>(this->data_ptr)) {
    return std::get<float *>(this->data_ptr)[total_offset];
  } else if (std::holds_alternative<void *>(this->data_ptr)) {
    // return std::get<void *>(this->data_ptr)[offset];
  }
  return -1;
}

// TODO: fix the type float for value and make it dynamic
template <typename... Args>
void Tensor::setElement(float value, Args... indexes) {
  int indices[] = {indexes...};
  this->throw_out_of_bound(indices);
  int offset = this->_compute_offset(indices);
  if (std::holds_alternative<int *>(this->data_ptr)) {
    std::get<int *>(this->data_ptr)[offset] = value;
  } else if (std::holds_alternative<float *>(this->data_ptr)) {
    std::get<float *>(this->data_ptr)[offset] = value;
  } else if (std::holds_alternative<void *>(this->data_ptr)) {
    // std::get<int *>(this->data_ptr)[offset] = value;
  }
}

// TODO: impelement this
Tensor Tensor::transpose() const { throw std::logic_error("not implemented"); }
void Tensor::print(int dim, int offset) const {
  std::string builder;
  builder.append("Tensor(");
  this->tensor__repr__(0, 0, 0, builder);
  builder.append(", dtype=" + getTypeName(this->dtype));
  builder.append(")");
  std::cout << builder << "\n";
  return;
}

std::string Tensor::__repr__() const {
  std::string builder;
  builder.append("Tensor(");
  this->tensor__repr__(0, 0, 0, builder);
  builder.append(", dtype=" + getTypeName(this->dtype));
  builder.append(", requires_grad=" +
                 std::string((this->requires_grad ? "True" : "False")));
  builder.append(")");
  return builder;
}

void Tensor::tensor__repr__(int depth, int offset, int indent,
                            std::string &builder) const {
  int k = 3;
  for (int i = 0; i < indent; ++i)
    builder.append(" ");
  builder.append("[");

  if (depth == this->ndim - 1) {
    for (int i = 0; i < this->dims[depth]; ++i) {
      if (i == k && this->dims[depth] > 3 * k) {
        builder.append("... ");
        i = this->dims[depth] - k;
      }
      int index = offset + i * this->stride[depth];
      if (this->dtype == DType::float16 || this->dtype == DType::float32) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.6e", this->_get_element(index));
        builder.append(buffer);
      } else {
        builder.append(std::to_string(this->_get_element(index)));
      }
      if (i < this->dims[depth] - 1) {
        builder.append(", ");
      }
    }
    builder.append("]");

  } else {
    builder.append("\n");
    for (int i = 0; i < this->dims[depth]; ++i) {
      if (i == k && this->dims[depth] > 3 * k) {
        for (int j = 0; j < indent + 1; ++j)
          builder.append(" ");
        builder.append("...\n");
        i = this->dims[depth] - k;
      }

      tensor__repr__(depth + 1, offset + i * this->stride[depth], indent + 1,
                     builder);

      if (i < this->dims[depth] - 1) {
        builder.append(",\n");
      }
    }

    builder.append("\n");
    for (int i = 0; i < indent; ++i)
      builder.append(" ");
    builder.append("]");
  }
}

void Tensor::print_buffer() const {
  for (int i = 0; i < this->memory->size; i++) {
    std::cout << this->_get_element(i) << " ";
  }
  std::cout << std::endl;
}

Tensor *Tensor::execute_broadcastable_operation(OPType op, Tensor *other,
                                                bool inplace) {
  if (this->requires_grad || other->requires_grad) {
    this->requires_grad = other->requires_grad = true;
  }
  if (inplace) {
    if (this->requires_grad)
      return NULL;
    dispatcher->call(op, this->device, this, other, this);
    return this;
  }
  auto result_shape = compute_broadcast_shape(this, other);
  std::shared_ptr<Memory> result_memory = pool->request_memory(
      this->device,
      std::accumulate(result_shape.begin(), result_shape.end(), 1,
                      std::multiplies<int>()),
      this->dtype);

  Tensor *result =
      new Tensor(result_memory, result_shape, this->dtype, this->requires_grad);
  dispatcher->call(op, this->device, this, other, result);
  return result;
}

Tensor *Tensor::execute_init_operation(OPType op, std::vector<int> shape,
                                       DType dtype, bool requires_grad,
                                       DeviceType device) {
  std::shared_ptr<Memory> result_memory = pool->request_memory(
      device,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
          getDTypeSize(dtype),
      dtype);
  Tensor *result = new Tensor(result_memory, shape, dtype, requires_grad);
  dispatcher->call(op, device, result, nullptr, nullptr);
  return result;
}

Tensor *Tensor::execute_binary_operation(OPType op, Tensor *other) {
  if (this->requires_grad || other->requires_grad) {
    this->requires_grad = other->requires_grad = true;
  }
  std::shared_ptr<Memory> result_memory =
      pool->request_memory(this->device,
                           std::accumulate(this->dims.begin(), this->dims.end(),
                                           1, std::multiplies<int>()),
                           this->dtype);

  Tensor *result =
      new Tensor(result_memory, this->dims, this->dtype, this->requires_grad);
  dispatcher->call(op, this->device, this, other, result);
  return result;
}

bool Tensor::all() {
  bool allTrue = true;
  for (int i = 0; i < this->size; i++) {
    if (false == this->_get_element(i)) {
      allTrue = false;
    }
  }
  return allTrue;
}
bool Tensor::any() {
  bool anyTrue = false;
  for (int i = 0; i < this->size; i++) {
    if (this->_get_element(i)) {
      anyTrue = true;
    }
  }
  return anyTrue;
}

// ================================================================================================================================
// Arithemetic
// ================================================================================================================================
Tensor *Tensor::add(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::ADD, other, inplace);
}
Tensor *Tensor::sub(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::SUB, other, inplace);
}

Tensor *Tensor::mul(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::MUL, other, inplace);
}

Tensor *Tensor::div(Tensor *other, bool inplace) {
  // TODO: fix this division by zero checking
  Tensor *zeros = Tensor::zeros(other->dims);
  if (other->logical_e(zeros)->any()) {
    throw std::runtime_error("division by zero");
  }
  return execute_broadcastable_operation(OPType::DIV, other, inplace);
}

// Comparison operators
Tensor *Tensor::logical_e(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape constraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_E, other);
}
Tensor *Tensor::logical_ne(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape constraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_NE, other);
}
Tensor *Tensor::logical_gt(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape constraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_GT, other);
}

Tensor *Tensor::logical_gte(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_GTE, other);
}

Tensor *Tensor::logical_lt(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_LT, other);
}

Tensor *Tensor::logical_lte(Tensor *other) {
  if (this->dims != other->dims) {
    throw std::runtime_error("shape contraint failed");
  }
  return this->execute_binary_operation(OPType::LOGICAL_LTE, other);
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
Tensor *Tensor::ones(std::vector<int> shape, DType dtype, bool requires_grad) {
  return Tensor::execute_init_operation(OPType::ONES_INIT, shape, dtype,
                                        requires_grad);
}

Tensor *Tensor::zeros(std::vector<int> shape, DType dtype, bool requires_grad) {
  return Tensor::execute_init_operation(OPType::ZEROES_INIT, shape, dtype,
                                        requires_grad);
}

Tensor *Tensor::eye(int n, DType dtype, bool requires_grad) {
  std::vector<int> shape = {n, n};
  return Tensor::execute_init_operation(OPType::EYE_INIT, shape, dtype,
                                        requires_grad);
}

/*
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

Tensor Tensor::randint(std::vector<int> shape, int min, int max, DType dtype)
{ id<MTLBuffer> meta = device_mps->createBuffer(shape.data(), shape.size(),
dtype);

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
