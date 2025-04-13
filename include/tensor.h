#pragma once

#include <tuple>
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include <Metal/Metal.h>
#include <sys/types.h>
#include <vector>
class Tensor {
private:
  id<MTLBuffer> storage;
  std::vector<int> stride;
  float *data_ptr;
  bool requires_grad;
  int ndim;
  void _compte_stride();
  int _compute_offset(std::vector<int> indexes) const;
  std::vector<int> _compute_broadcast_shape(const Tensor *other) const;

  int _compute_broadcast_index(int flat_index,
                               const std::vector<int> &source_shape,
                               const std::vector<int> &target_shape) const;
  void throw_out_of_bound(std::vector<int> indexes) const;

  Tensor _dispatch_kernel_operation(const Tensor *other,
                                    std::string kernel_function) const;

  Tensor _dispatch_kernel_operation_inplace(const Tensor *other,
                                            std::string kernel_function);

public:
  std::vector<int> dims;
  int size;
  Tensor(std::vector<int> dims, bool requires_grad = false);
  Tensor(id<MTLBuffer> buffer, std::vector<int> dims,
         bool requires_grad = false);

  Tensor(std::vector<float> &values, std::vector<int> dims,
         bool requires_grad = false);
  template <typename T>
  Tensor(std::vector<T> &values, std::vector<int> dims,
         bool requires_grad = false);

  // initialization methods
  static Tensor ones(std::vector<int> shape, std::string dtype = "float");
  static Tensor zeros(std::vector<int> shape, std::string dtype = "float");
  static Tensor eye(int n, std::string dtype = "float");
  static Tensor empty(std::vector<int> shape, std::string dtype = "float");

  template <typename T>
  static Tensor full(std::vector<int> shape, T n, std::string dtype = "float");
  static Tensor clone(Tensor *other, std::string dtype = "float");
  static Tensor rand(std::vector<int> shape, std::string dtype = "float");
  static Tensor randn(std::vector<int> shape, std::string dtype = "float");
  static Tensor normal(std::vector<int> shape, float mean = 0, float stddev = 1,
                       std::string dtype = "float");
  static Tensor randint(std::vector<int> shape, int min, int max,
                        std::string dtype = "float");
  static Tensor poission(Tensor &other, std::string dtype = "float");
  static Tensor bernoulli(Tensor &other, std::string dtype = "float");

  // getters & setters
  std::vector<int> strides();
  template <typename... Args> double getElement(Args... indexes) const;
  template <typename... Args> void setElement(float value, Args... indexes);

  // arithmetic operators
  Tensor add(const Tensor *other, bool inplace);
  Tensor sub(const Tensor *other, bool inplace);
  Tensor mul(const Tensor *other, bool inplace);
  Tensor div(const Tensor *other, bool inplace);
  Tensor matmul(const Tensor *other) const;
  Tensor pow(float exp, bool inplace);

  // Comparison operators
  Tensor logical_e(const Tensor *other) const;
  Tensor logical_ne(const Tensor *other) const;
  Tensor logical_gt(const Tensor *other) const;
  Tensor logical_gte(const Tensor *other) const;
  Tensor logical_lt(const Tensor *other) const;
  Tensor logical_lte(const Tensor *other) const;

  // Mathematical operations
  Tensor exp(bool inplace);
  Tensor log(bool inplace);

  // TODO: modify this to have a numpy like behaviour
  bool all();
  bool any();
  Tensor sqrt(bool inplace);

  // Utility methods
  Tensor transpose() const;

  // Input/Output
  void print() const;
  void print_matrix() const;
};
