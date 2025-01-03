#include "mps.h"
#include <MacTypes.h>
#include <Metal/Metal.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sys/types.h>
#include <vector>

// =====================================================================
//                            DEBUG UTILITY
// =====================================================================
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << static_cast<T>(vec[i]);
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
// =====================================================================

// =====================================================================
//                          UTILITY
// =====================================================================
template <typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  // Check if sizes are equal
  if (lhs.size() != rhs.size()) {
    return false;
  }

  // Compare each element
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
}

MPS *device_mps = new MPS();
// =====================================================================

template <typename T> class Tensor {
private:
  // =====================================================================
  id<MTLBuffer> storage;
  std::vector<int> stride;
  T *data_ptr;
  int ndim;
  // =====================================================================

  void _compte_stride() {
    /*strides[i] = (j=i+1 ∏ len(dims) - 1){shape[j]}*/
    int value = 1;
    this->stride.push_back(value);
    for (uint i = this->ndim - 1; i > 0; i--) {
      value *= this->dims[i];
      this->stride.insert(this->stride.begin(), value);
    }
  }

  int _compute_offset(std::vector<int> indexes) const {
    int n = indexes.size();
    int offset = 0;
    assert(n == this->stride.size());
    for (int i = 0; i < n; i++) {
      offset += indexes[i] * this->stride[i];
    }
    return offset;
  }

  void throw_out_of_bound(std::vector<int> indexes) const {
    for (int i = 0; i < indexes.size(); i++) {
      if (indexes[i] >= this->dims[i]) {
        throw std::out_of_range("");
      }
    }
  }

  Tensor _dispatch_kernel_operation(const Tensor *other,
                                    std::string kernel_function) const {
    std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
    id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
    id<MTLBuffer> result;
    result = device_mps->createEmptyBuffer<T>(this->size);
    device_mps->execute_kernel_binary(kernel_function, this->storage,
                                      other->storage, result, meta);
    return Tensor(result, this->dims);
  }
  Tensor _dispatch_kernel_operation_inplace(const Tensor *other,
                                            std::string kernel_function) {
    std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
    id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
    device_mps->execute_kernel_binary(kernel_function, this->storage,
                                      other->storage, this->storage, meta);
    return *this;
  }
  // =====================================================================

public:
  // =====================================================================
  std::vector<int> dims;
  int size;

  // =====================================================================
  Tensor(std::vector<int> dims) {
    this->size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    this->dims = dims;
    this->ndim = dims.size();
    this->storage = device_mps->createEmptyBuffer<T>(size);
    this->data_ptr = (T *)[this->storage contents];
    this->_compte_stride();
  }

  Tensor(id<MTLBuffer> buffer, std::vector<int> dims) {
    this->storage = buffer;
    this->data_ptr = (T *)[this->storage contents];
    this->dims = dims;
    this->ndim = dims.size();
    this->_compte_stride();
    this->size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  }

  Tensor(std::vector<T> &values, std::vector<int> dims) {
    this->storage = device_mps->createBuffer(values.data(), values.size());
    this->data_ptr = (T *)[this->storage contents];
    this->dims = dims;
    this->ndim = dims.size();
    this->_compte_stride();
    this->size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  }

  // Destructor
  ~Tensor() {
    // Body for destructor
  }

  // =====================================================================
  // Accessors
  std::vector<int> strides() { return this->stride; }
  template <typename... Args> double getElement(Args... indexes) const {
    std::vector<int> indices = {indexes...};
    this->throw_out_of_bound(indices);
    int offset = this->_compute_offset(indices);
    return this->data_ptr[offset];
  }
  template <typename... Args> void setElement(T value, Args... indexes) {
    int indices[] = {indexes...};
    this->throw_out_of_bound(indices);
    int offset = this->_compute_offset(indices);
    this->data_ptr[offset] = value;
  }

  // arithemetic Operators
  Tensor add(const Tensor *other, bool inplace) {
    assert(this->dims == other->dims && this->dims.size() == 2);
    return inplace
               ? this->_dispatch_kernel_operation_inplace(other, "add_matrix")
               : this->_dispatch_kernel_operation(other, "add_matrix");
  }
  Tensor subtract(const Tensor *other, bool inplace) {
    assert(this->dims == other->dims && this->dims.size() == 2);
    return inplace ? this->_dispatch_kernel_operation_inplace(other,
                                                              "subtract_matrix")
                   : this->_dispatch_kernel_operation(other, "subtract_matrix");
  }

  Tensor elementwise_multiply(const Tensor *other, bool inplace) {
    assert(this->dims == other->dims && this->dims.size() == 2);
    return inplace ? this->_dispatch_kernel_operation_inplace(
                         other, "elementwise_multiply_matrix")
                   : this->_dispatch_kernel_operation(
                         other, "elementwise_multiply_matrix");
  }
  Tensor elementwise_divide(const Tensor *other, bool inplace) {
    assert(this->dims == other->dims && this->dims.size() == 2);
    return inplace ? this->_dispatch_kernel_operation_inplace(
                         other, "elementwise_divide_matrix")
                   : this->_dispatch_kernel_operation(
                         other, "elementwise_divide_matrix");
  }
  Tensor matrix_multiply(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    std::vector<int> m = {this->dims[0], this->dims[1], other->dims[1]};
    id<MTLBuffer> meta = device_mps->createBuffer(m.data(), 3);
    id<MTLBuffer> result;
    result = device_mps->createEmptyBuffer<T>(this->size);
    device_mps->execute_kernel_binary("matrix_multiply", this->storage,
                                      other->storage, result, meta);
    return Tensor(result, this->dims);
  }

  Tensor pow(float exp, bool inplace) {
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
  Tensor logical_e(const Tensor *other) const {
    assert(this->dims == other->dims && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_e");
  }
  Tensor logical_ne(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_ne");
  }
  Tensor logical_gt(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_gt");
  }

  Tensor logical_gte(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_gte");
  }

  Tensor logical_lt(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_lt");
  }

  Tensor logical_lte(const Tensor *other) const {
    assert(this->dims[1] == other->dims[0] && this->dims.size() == 2);
    return this->_dispatch_kernel_operation(other, "logical_lte");
  }

  // Mathematical operations
  Tensor exp(bool inplace) {
    id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 2);
    id<MTLBuffer> result;
    if (!inplace) {
      result = device_mps->createEmptyBuffer<T>(this->size);
      std::cout << result.length << std::endl;
      device_mps->execute_kernel_unary("exp", this->storage, result, meta);
    } else {
      device_mps->execute_kernel_unary("exp", this->storage, this->storage,
                                       meta);
    }
    return inplace ? *this : Tensor(result, this->dims);
  }

  Tensor log(bool inplace) {
    id<MTLBuffer> meta = device_mps->createBuffer(this->dims.data(), 2);
    id<MTLBuffer> result;
    if (!inplace) {
      result = device_mps->createEmptyBuffer<T>(this->size);
      std::cout << result.length << std::endl;
      device_mps->execute_kernel_unary("log", this->storage, result, meta);
    } else {
      device_mps->execute_kernel_unary("log", this->storage, this->storage,
                                       meta);
    }
    return inplace ? *this : Tensor(result, this->dims);
  }

  // Utility methods
  Tensor transpose() const {
    // Body for transpose
  }

  // Input/Output
  void print() const {
    T *ptr = (T *)[this->storage contents];
    std::cout << ptr[0] << std::endl;
  }
  void print_matrix() const {
    assert(this->stride.size() == 2);
    for (int i = 0; i < this->dims[0]; i++) {
      for (int j = 0; j < this->dims[1]; j++) {
        std::cout << this->data_ptr[this->stride[0] * i + this->stride[1] * j]
                  << " ";
      }
      std::cout << std::endl;
    }
  }
};

int main() {
  std::vector<float> data2 = {1.2, 2.3, 3.6, 4.0, 5.9, 6.1, 7.4, 8.2, 9.3};
  std::vector<uint> conf = {3, 3, 3};
  std::vector<float> resul(9, 0);
  std::vector<float> data1 = {2.3, 3.6, 4.0, 5.9, 6.1, 7.4, 8.2, 9.3, 1.2};
  Tensor<float> *mat_a = new Tensor<float>(data1, std::vector<int>{3, 3});
  mat_a->print_matrix();
  std::cout << std::endl;
  Tensor<float> *mat_b = new Tensor<float>(data2, std::vector<int>{3, 3});
  mat_b->print_matrix();
  Tensor<float> result = mat_a->log(false);
  std::cout << std::endl;
  result.print_matrix();
  return 0;
}
