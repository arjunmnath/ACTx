#pragma once

#include "device.h"
#include "storage.h"
#include "types.h"
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
class MPS : Device {
private:
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> commandQueue;
  id<MTLHeap> heap;

  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::string name = "mps";

public:
  MPS();
  void _init_pipeline(std::string metal_function_name);
  std::pair<size_t, size_t> compute_threads(size_t N, size_t maxTPG);

  void execute_kernel_nullary(std::string func, id<MTLBuffer> A,
                              id<MTLBuffer> meta, int N);
  void execute_kernel_unary(std::string func, id<MTLBuffer> input,
                            id<MTLBuffer> output, id<MTLBuffer> metadata,
                            int N);
  void execute_kernel_binary(std::string func, id<MTLBuffer> A, id<MTLBuffer> B,
                             id<MTLBuffer> result, id<MTLBuffer> meta, int N,
                             int offset_a = 0, int offset_b = 0,
                             int offset_result = 0);

  void initiate_dispatch_nullary(std::string kernel_method, Tensor *input);

  void initiate_dispatch_unary(std::string kernel_method, const Tensor *input,
                               Tensor *output);

  void initiate_dispatch_binary(std::string kernel_method, const Tensor *a,
                                const Tensor *b, Tensor *result);

  void createEmptyBuffer(int bytesize, DType type, Storage *storage);
  id<MTLBuffer> clone(id<MTLBuffer> buffer);
  void copy_vector_to_buffer(void *ptr, Memory &memory, int buffer_size);

  // arithmetic kernels
  void negate(Tensor *input, Tensor *output);
  void add(const Tensor *a, const Tensor *b, Tensor *result) override;
  void sub(const Tensor *a, const Tensor *b, Tensor *result) override;
  void mul(const Tensor *a, const Tensor *b, Tensor *result) override;
  void div(const Tensor *a, const Tensor *b, Tensor *result) override;
  void pow(const Tensor *a, const Tensor *b, Tensor *result) override;

  // init kernels
  void ones(Tensor *a) override;
  void zeros(Tensor *a) override;
  void eye(Tensor *a) override;
  void full(Tensor *n, Tensor *result) override;
  void rand(Tensor *a, Tensor *meta) override;

  // comparison
  void logical_e(const Tensor *a, const Tensor *b, Tensor *result) override;
  void logical_ne(const Tensor *a, const Tensor *b, Tensor *result) override;
  void logical_gt(const Tensor *a, const Tensor *b, Tensor *result) override;
  void logical_gte(const Tensor *a, const Tensor *b, Tensor *result) override;
  void logical_lt(const Tensor *a, const Tensor *b, Tensor *result) override;
  void logical_lte(const Tensor *a, const Tensor *b, Tensor *result) override;

  // math functions
  void sqrt(const Tensor *input, Tensor *output) override;
  void exp(const Tensor *input, Tensor *output) override;
  void log(const Tensor *input, Tensor *output) override;
  void log10(const Tensor *input, Tensor *output) override;
  void log2(const Tensor *input, Tensor *output) override;

  // trignometric
  void sin(const Tensor *input, Tensor *output) override;
  void cos(const Tensor *input, Tensor *output) override;
  void tan(const Tensor *input, Tensor *output) override;
  void asin(const Tensor *input, Tensor *output) override;
  void acos(const Tensor *input, Tensor *output) override;
  void atan(const Tensor *input, Tensor *output) override;
  void atan2(const Tensor *x, const Tensor *y, Tensor *output) override;

  // hyperbolic
  void sinh(const Tensor *input, Tensor *output) override;
  void cosh(const Tensor *input, Tensor *output) override;
  void tanh(const Tensor *input, Tensor *output) override;
  void asinh(const Tensor *input, Tensor *output) override;
  void acosh(const Tensor *input, Tensor *output) override;
  void atanh(const Tensor *input, Tensor *output) override;

  // not implemented
  void matmul(const Tensor *a, const Tensor *b, Tensor *result) override;
};
#endif
