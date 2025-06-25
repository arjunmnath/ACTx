#pragma once

#include "memory.h"
#include "op_types.h"
#include "types.h"
#include <sys/types.h>
#include <tuple>
#include <variant>
#include <vector>

class OpNode;
struct Slice {
  int start;
  int stop;
  int step;
  Slice(int s = 0, int e = -1, int st = 1) : start(s), stop(e), step(st) {}
};
class Tensor {
private:
  bool is_view = false;
  int offset_elements;
  std::variant<void *, float *, int *> data_ptr;
  void _compte_stride();
  int _compute_offset(std::vector<int> indexes) const;
  void reinterpret_pointer(void *ptr);
  int _compute_broadcast_index(int flat_index,
                               const std::vector<int> &source_shape,
                               const std::vector<int> &target_shape) const;

  void throw_out_of_bound(std::vector<int> indexes) const;
  Tensor *execute_broadcastable_operation(OPType op, Tensor *other,
                                          bool inplace);
  Tensor *execute_binary_operation(OPType op, Tensor *other);

  // TODO: change default devicetype to cpu
  static Tensor *execute_init_operation(OPType op, std::vector<int> shape,
                                        DType dtype, bool requires_grad,
                                        DeviceType device);
  std::vector<OpNode *> topo_sort();

public:
  int ndim;
  Tensor *grad = nullptr;
  OpNode *node = nullptr;
  std::vector<int> dims;
  std::vector<int> stride;
  bool requires_grad;
  DType dtype;
  size_t size;
  DeviceType device;
  Memory *memory;
  bool is_contigous = true;
  bool is_leaf = this->node == nullptr;
  // TODO: change default devicetype to cpu
  Tensor(std::vector<int> dims, DType dtype = DType::float32,
         bool requires_grad = false, DeviceType device = DeviceType::MPS);
  Tensor(Memory *memory, std::vector<int> dims, DType dtype = DType::float32,
         bool requires_grad = false, DeviceType device = DeviceType::MPS);

  Tensor(std::vector<float> &values, std::vector<int> dims,
         DType dtype = DType::float32, bool requires_grad = false,
         DeviceType device = DeviceType::MPS);
  // template <typename T>
  // Tensor(std::vector<T> &values, std::vector<int> dims,
  //        DType dtype = DType::float32, bool requires_grad = false);
  //
  //
  // FIX: template type
  float _get_element(int offset) const;
  int offset() const;

  // initialization methods
  static Tensor *ones(std::vector<int> shape, DType dtype = DType::float32,
                      bool requires_grad = false,
                      DeviceType device = DeviceType::MPS);
  static Tensor *zeros(std::vector<int> shape, DType dtype = DType::float32,
                       bool requires_grad = false,
                       DeviceType device = DeviceType::MPS);
  static Tensor *eye(int n, DType dtype = DType::float32,
                     bool requires_grad = false,
                     DeviceType device = DeviceType::MPS);
  static Tensor *empty(std::vector<int> shape, DType dtype = DType::float32,
                       bool requires_grad = false,
                       DeviceType device = DeviceType::MPS);

  static Tensor *full(std::vector<int> shape, float n,
                      DType dtype = DType::float32, bool requires_grad = false,
                      DeviceType device = DeviceType::MPS);

  static Tensor *empty_like(Tensor *a);
  static Tensor *ones_like(Tensor *a);
  static Tensor *zeros_like(Tensor *a);
  static Tensor *full_like(Tensor *a, float n);
  static Tensor *clone(Tensor *other);

  // arithmetic operators
  Tensor *negate(bool inplace = false);
  Tensor *add(Tensor *other, bool inplace = false);
  Tensor *sub(Tensor *other, bool inplace = false);
  Tensor *mul(Tensor *other, bool inplace = false);
  Tensor *div(Tensor *other, bool inplace = false);
  Tensor *pow(float exp, bool inplace = false);
  Tensor *matmul(Tensor *other) const;

  // Comparison operators
  Tensor *logical_e(Tensor *other);
  Tensor *logical_ne(Tensor *other);
  Tensor *logical_gt(Tensor *other);
  Tensor *logical_gte(Tensor *other);
  Tensor *logical_lt(Tensor *other);
  Tensor *logical_lte(Tensor *other);

  // Mathematical operations
  Tensor *exp(bool inplace = false);
  Tensor *sqrt(bool inplace = false);
  Tensor *log(bool inplace = false);
  Tensor *log10(bool inplace = false);
  Tensor *log2(bool inplace = false);

  // trignometric operations
  Tensor *sin(bool inplace = false);
  Tensor *cos(bool inplace = false);
  Tensor *tan(bool inplace = false);
  Tensor *atan(bool inplace = false);
  Tensor *acos(bool inplace = false);
  Tensor *asin(bool inplace = false);
  Tensor *atan2(Tensor *other, bool inplace = false);

  // hyperbolic
  Tensor *sinh(bool inplace = false);
  Tensor *cosh(bool inplace = false);
  Tensor *tanh(bool inplace = false);
  Tensor *atanh(bool inplace = false);
  Tensor *acosh(bool inplace = false);
  Tensor *asinh(bool inplace = false);

  // not implemented
  static Tensor *rand(std::vector<int> shape, DType dtype);
  static Tensor *randn(std::vector<int> shape, DType dtype = DType::float32);
  static Tensor *normal(std::vector<int> shape, float mean = 0,
                        float stddev = 1, DType dtype = DType::float32);
  static Tensor *randint(std::vector<int> shape, int min, int max,
                         DType dtype = DType::float32);
  static Tensor *poission(Tensor &other, DType dtype = DType::float32);
  static Tensor *bernoulli(Tensor &other, DType dtype = DType::float32);

  // TODO: modify this to have a numpy like behaviour
  bool all();
  bool any();

  // Utility methods
  Tensor transpose() const;
  Tensor *view(std::vector<Slice> &slices) const;

  void backward();
  void detach();
  // Input/Output

  void print(int dim = 0, int offset = 0) const;
  void print_buffer() const;
  std::string __repr__() const;
  void tensor__repr__(int depth, int offset, int indent,
                      std::string &builder) const;
  // getters & setters
  std::vector<int> strides();
  template <typename... Args> void setElement(float value, Args... indexes);
  template <typename... Args> double getElement(Args... indexes) const {
    std::vector<int> indices = {indexes...};
    this->throw_out_of_bound(indices);
    int offset = this->_compute_offset(indices);
    if (std::holds_alternative<int *>(this->data_ptr)) {
      return std::get<int *>(this->data_ptr)[offset];
    } else if (std::holds_alternative<float *>(this->data_ptr)) {
      return std::get<float *>(this->data_ptr)[offset];
    } else if (std::holds_alternative<void *>(this->data_ptr)) {
      // return std::get<void *>(this->data_ptr)[offset];
    }
    return -1;
  }
};
