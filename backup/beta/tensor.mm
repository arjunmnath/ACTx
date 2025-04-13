#include "cpu.cpp"
#include "mps_helper.h"
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
// =====================================================================

template <typename T> class Tensor {
private:
  // =====================================================================
  // private members
  // =====================================================================
  T *data_ptr;
  std::vector<int> stride;
  int ndim;
  Device *device;
  // =====================================================================

  // =====================================================================
  // private methods
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
  // =====================================================================
public:
  // public properties
  std::vector<int> dims;
  int size;
  using dtype = T;
  // Constructors
  Tensor(size_t rows, size_t cols) {
    // Body to initialize with dimensions
    this->data_ptr = new T[rows * cols];
  }
  Tensor(std::vector<T> &values, std::vector<int> dims) {
    // Body to initialize with a 2D vector
    this->data_ptr = values.data();
    this->dims = dims;
    this->ndim = dims.size();
    this->_compte_stride();
    this->size =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    this->device = new CPUHelper();
  }

  // Destructor
  ~Tensor() {
    // Body for destructor
  }

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
    int offset = indices[0];
  }

  // Operators
  Tensor operator+(const Tensor &other) const {
    // Body for addition
  }
  Tensor operator-(const Tensor &other) const {
    // Body for subtraction
  }
  Tensor elementwiseMultiply(const Tensor &other) const {
    // Body for element-wise multiplication
    assert(this->dims == other->dims);
  }
  Tensor matrixMultiply(const Tensor &other) const {
    // Body for matrix multiplication
  }
  Tensor operator*(double scalar) const {
    // Body for scalar multiplication
  }

  // Comparison operators
  bool operator==(const Tensor &other) const {
    // Body for equality check
  }
  bool operator!=(const Tensor &other) const {
    // Body for inequality check
  }

  // Utility methods
  Tensor transpose() const {
    // Body for transpose
  }
  Tensor inverse() const {
    // Body for inverse (square matrices only)
  }
  double determinant() const {
    // Body for determinant (square matrices only)
  }

  // Input/Output
  void print() const {}
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

  void printType() {
    std::cout << "Tensor type: " << typeid(dtype).name() << std::endl;
  }
};

int main() {
  std::vector<float> data2 = {1.2, 2.3, 3.6, 4.0, 5.9, 6.1, 7.4, 8.2, 9.3};
  std::vector<uint> conf = {3, 3, 3};
  std::vector<float> resul(9, 0);
  std::vector<float> data1 = {2.3, 3.6, 4.0, 5.9, 6.1, 7.4, 8.2, 9.3, 1.2};

  /*
  id<MTLBuffer> a = device_mps->createBuffer(data1.data(), data1.size());
  id<MTLBuffer> b = device_mps->createBuffer(data2.data(), data2.size());
  id<MTLBuffer> result = device_mps->createBuffer(resul.data(), resul.size());
  id<MTLBuffer> meta = device_mps->createBuffer(conf.data(), conf.size());
  device_mps->execute_kernel("add_matrix", a, b, result, meta);
  uint stride[] = {3, 1};
  device_mps->print_buffer_contents(
      std::vector<id<MTLBuffer>>{a, b, result, meta}, stride);
  */

  Tensor<float> *mat_a = new Tensor<float>(data1, std::vector<int>{3, 3});
  Tensor<float> *mat_b = new Tensor<float>(data2, std::vector<int>{3, 3});
  mat_a->device->add(data1.data(), data2.data());
  return 0;
}
