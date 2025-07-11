#include "tensor.h"
#include "main.h"
#include "opnode.h"
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
#include <unordered_set>
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
  view_tensor->size =
      std::accumulate(view_tensor->dims.begin(), view_tensor->dims.end(), 1,
                      std::multiplies<int>());
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
Tensor::Tensor(std::vector<int> dims, DType dtype, bool requires_grad,
               DeviceType device) {
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  this->dims = dims;
  this->ndim = dims.size();
  this->device = device;
  this->dtype = dtype;
  this->memory = pool->request_memory(this->device, this->size, this->dtype);
  this->offset_elements = 0;
  this->reinterpret_pointer(this->memory->data_ptr);
  this->_compte_stride();
  this->requires_grad = requires_grad;
  if (requires_grad) {
    this->node = new OpNode;
    this->node->type = OPType::NO_OP;
  }
}

Tensor::Tensor(Memory *memory, std::vector<int> dims, DType dtype,
               bool requires_grad, DeviceType device) {
  this->dims = dims;
  this->memory = memory;
  this->dtype = dtype;
  this->reinterpret_pointer(this->memory->data_ptr);
  this->device = device;
  this->ndim = dims.size();

  this->offset_elements = 0;
  this->_compte_stride();
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  this->requires_grad = requires_grad;
  if (requires_grad) {
    this->node = new OpNode;
    this->node->type = OPType::NO_OP;
  }
}

// FIX: fix the vector<float> and dtype mismatch and allocate memory and do
// memcpy
Tensor::Tensor(std::vector<float> &values, std::vector<int> dims, DType dtype,
               bool requires_grad, DeviceType device) {
  if (values.size() == 0) {
    throw std::runtime_error("values expected");
  }
  this->dtype = dtype;
  this->dims = dims;
  this->ndim = dims.size();
  this->_compte_stride();
  this->offset_elements = 0;
  this->device = device;
  this->size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  assert(values.size() == this->size);
  this->memory = pool->request_memory(this->device, this->size, this->dtype);
  mps->copy_vector_to_buffer(values.data(), *this->memory,
                             values.size() * getDTypeSize(dtype));
  this->reinterpret_pointer(this->memory->data_ptr);
  this->requires_grad = requires_grad;
  if (requires_grad) {
    this->node = new OpNode;
    this->node->type = OPType::NO_OP;
  }
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
  builder.append(", requires_grad=" +
                 std::string((this->requires_grad ? "True" : "False")));
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
  for (int i = 0; i < this->size; i++) {
    std::cout << this->_get_element(i) << " ";
  }
  std::cout << std::endl;
}

int Tensor::offset() const { return this->offset_elements; }
Tensor *Tensor::execute_broadcastable_operation(OPType op, Tensor *other,
                                                bool inplace) {
  if (inplace) {
    // TODO: recheck this return null logic
    if (this->requires_grad && other->requires_grad)
      return NULL;
    dispatcher->call(op, this->device, {this, other, this});
    return this;
  }
  auto result_shape = compute_broadcast_shape(this, other);
  Memory *result_memory = pool->request_memory(
      this->device,
      std::accumulate(result_shape.begin(), result_shape.end(), 1,
                      std::multiplies<int>()),
      this->dtype);

  Tensor *result = new Tensor(result_memory, result_shape, this->dtype,
                              this->requires_grad || other->requires_grad);
  dispatcher->call(op, this->device, {this, other, result});
  return result;
}

Tensor *Tensor::execute_init_operation(OPType op, std::vector<int> shape,
                                       DType dtype, bool requires_grad,
                                       DeviceType device) {
  Memory *result_memory = pool->request_memory(
      device,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()),
      dtype);
  Tensor *result =
      new Tensor(result_memory, shape, dtype, requires_grad, device);
  dispatcher->call(op, device, {result});
  return result;
}

Tensor *Tensor::execute_binary_operation(OPType op, Tensor *other) {
  Memory *result_memory =
      pool->request_memory(this->device,
                           std::accumulate(this->dims.begin(), this->dims.end(),
                                           1, std::multiplies<int>()),
                           this->dtype);

  Tensor *result = new Tensor(result_memory, this->dims, this->dtype,
                              this->requires_grad || other->requires_grad);
  dispatcher->call(op, this->device, {this, other, result});
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

std::vector<OpNode *> Tensor::topo_sort() {
  std::vector<OpNode *> topo;
  std::unordered_set<OpNode *> visited;

  std::function<void(OpNode *)> dfs = [&](OpNode *node) {
    if (visited.find(node) == visited.end()) {
      visited.insert(node);
      for (Tensor *parent : node->inputs) {
        if (parent->node)
          dfs(parent->node);
      }
      topo.push_back(node);
    }
  };
  dfs(this->node);
  return topo;
}

void Tensor::backward() {
  if (!this->requires_grad)
    return;
  this->grad = Tensor::ones(this->dims, this->dtype, false, this->device);

  std::vector<OpNode *> sorted = this->topo_sort();
  OpNode *current_node;
  for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
    current_node = *it;
    if ((*it)->type == OPType::NO_OP)
      continue;
    (*it)->op->backward(*it);
    // bool has_nonleaf = false;
    // for (Tensor *tensor = current_node->inputs.begin();
    //      tensor != current_node->inputs.end(); ++tensor) {
    //   if (tensor->requires_grad) {
    //     if (tensor->node->type == OPType::NO_OP) {
    //       Tensor *accumulated_grad = current_node->outputs[0]->grad;
    //       if (!tensor->grad) {
    //         tensor->grad = Tensor::clone(accumulated_grad);
    //       } else {
    //         tensor->grad->add(accumulated_grad, true);
    //       }
    //     } else {
    //       has_nonleaf = true;
    //     }
    //   }
    // }
    // if (has_nonleaf) {
    //   current_node->op->backward(current_node);
    // }
    // for (auto x : (*it)->outputs) {
    //   x->print();
    // }
  }
}
// ================================================================================================================================
// Arithemetic
// ================================================================================================================================
Tensor *Tensor::negate(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::NEGATE, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::NEGATE, this->device, {this, result});
    return result;
  }
}
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
  Tensor *zeros =
      Tensor::zeros(other->dims, other->dtype, false, other->device);
  free(zeros);
  return execute_broadcastable_operation(OPType::DIV, other, inplace);
}

Tensor *Tensor::pow(float exp, bool inplace) {
  std::vector<float> val = {exp};
  Tensor *other = new Tensor(val, {1});
  return execute_binary_operation(OPType::POW, other);
}

// Comparison operators
Tensor *Tensor::logical_e(Tensor *other) {
  return this->execute_binary_operation(OPType::LOGICAL_E, other);
}
Tensor *Tensor::logical_ne(Tensor *other) {
  return this->execute_binary_operation(OPType::LOGICAL_NE, other);
}
Tensor *Tensor::logical_gt(Tensor *other) {
  return this->execute_binary_operation(OPType::LOGICAL_GT, other);
}

Tensor *Tensor::logical_gte(Tensor *other) {
  return this->execute_binary_operation(OPType::LOGICAL_GTE, other);
}

Tensor *Tensor::logical_lt(Tensor *other) {
  return this->execute_binary_operation(OPType::LOGICAL_LT, other);
}

Tensor *Tensor::logical_lte(Tensor *other) {
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
*/

// Mathematical operations
Tensor *Tensor::exp(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::EXP, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::EXP, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::sqrt(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::SQRT, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::SQRT, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::log(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::LOG, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::LOG, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::log10(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::LOG10, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::LOG10, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::log2(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::LOG2, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::LOG2, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::sin(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::SIN, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::SIN, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::cos(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::COS, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::COS, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::tan(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::TAN, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::TAN, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::asin(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ASIN, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ASIN, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::acos(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ACOS, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ACOS, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::atan(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ATAN, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ATAN, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::atan2(Tensor *other, bool inplace) {
  return execute_broadcastable_operation(OPType::ATAN2, other, inplace);
}
Tensor *Tensor::sinh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::SINH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::SINH, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::cosh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::COSH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::COSH, this->device, {this, result});
    return result;
  }
}
Tensor *Tensor::tanh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::TANH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::TANH, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::asinh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ASINH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ASINH, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::acosh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ACOSH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ACOSH, this->device, {this, result});
    return result;
  }
}

Tensor *Tensor::atanh(bool inplace) {
  if (inplace) {
    dispatcher->call(OPType::ATANH, this->device, {this, this});
    return this;
  } else {
    Tensor *result =
        new Tensor(this->dims, this->dtype, this->requires_grad, this->device);
    dispatcher->call(OPType::ATANH, this->device, {this, result});
    return result;
  }
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
// =====================================================================================================================
Tensor *Tensor::empty(std::vector<int> shape, DType dtype, bool requires_grad,
                      DeviceType device) {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  Memory *new_buffer = pool->request_memory(device, size, dtype);
  Tensor *tensor = new Tensor(new_buffer, shape, dtype, requires_grad, device);
  return tensor;
}
Tensor *Tensor::ones(std::vector<int> shape, DType dtype, bool requires_grad,
                     DeviceType device) {
  return Tensor::execute_init_operation(OPType::ONES_INIT, shape, dtype,
                                        requires_grad, device);
}

Tensor *Tensor::zeros(std::vector<int> shape, DType dtype, bool requires_grad,
                      DeviceType device) {
  return Tensor::execute_init_operation(OPType::ZEROES_INIT, shape, dtype,
                                        requires_grad, device);
}

Tensor *Tensor::eye(int n, DType dtype, bool requires_grad, DeviceType device) {
  std::vector<int> shape = {n, n};
  return Tensor::execute_init_operation(OPType::EYE_INIT, shape, dtype,
                                        requires_grad, device);
}
// FIX: use varient for n
Tensor *Tensor::full(std::vector<int> shape, float n, DType dtype,
                     bool requires_grad, DeviceType device) {
  std::vector<float> val = {n};
  Tensor *other = new Tensor(val, {1});
  Memory *result_memory = pool->request_memory(
      device,
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()),
      dtype);

  Tensor *result = new Tensor(result_memory, shape, dtype, requires_grad);
  dispatcher->call(OPType::FULL_INIT, device, {other, result});
  free(other);
  return result;
}
Tensor *Tensor::empty_like(Tensor *a) {
  return Tensor::empty(a->dims, a->dtype, a->requires_grad, a->device);
}
Tensor *Tensor::ones_like(Tensor *a) {
  return Tensor::ones(a->dims, a->dtype, a->requires_grad, a->device);
}
Tensor *Tensor::zeros_like(Tensor *a) {
  return Tensor::zeros(a->dims, a->dtype, a->requires_grad, a->device);
}
Tensor *Tensor::full_like(Tensor *a, float n) {
  return Tensor::full(a->dims, n, a->dtype, a->requires_grad, a->device);
}

Tensor *Tensor::clone(Tensor *other) {
  Memory *new_buffer =
      pool->request_memory(other->device, other->size, other->dtype);
  Memory::copy(other->memory, new_buffer);
  Tensor *cloned = new Tensor(new_buffer, other->dims, other->dtype,
                              other->requires_grad, other->device);
  if (other->grad) {
    Memory *new_grad_buffer = pool->request_memory(
        other->grad->device, other->grad->memory->bytesize, other->grad->dtype);
    Memory::copy(other->grad->memory, new_grad_buffer);
    Tensor *grad_tensor =
        new Tensor(new_grad_buffer, other->grad->dims, other->grad->dtype,
                   other->grad->requires_grad, other->grad->device);
    cloned->grad = grad_tensor;
  }
  if (cloned->requires_grad) {
    cloned->node->outputs = {cloned};
    cloned->node->inputs = {other};
    cloned->node->type = OPType::CLONE;
    cloned->node->op = dispatcher->get(OPType::CLONE, cloned->device);
  }
  return cloned;
}

/*
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
