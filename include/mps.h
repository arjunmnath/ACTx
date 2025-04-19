#pragma once

#include "types.h"
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device.h"
#include <Metal/Metal.h>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
class MPS : Device {
private:
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> commandQueue;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::string name = "mps";

public:
  MPS();
  void _init_pipeline(std::string metal_function_name);
  void execute_kernel_binary(std::string func, id<MTLBuffer> A, id<MTLBuffer> B,
                             id<MTLBuffer> result, id<MTLBuffer> meta);
  void execute_kernel_binary_with_broadcast(
      std::string func, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> result,
      id<MTLBuffer> lshape, id<MTLBuffer> rshape, id<MTLBuffer> target,
      id<MTLBuffer> ranks);

  void execute_kernel_unary_with_broadcast(
      std::string func, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> result,
      id<MTLBuffer> lshape, id<MTLBuffer> rshape, id<MTLBuffer> target,
      id<MTLBuffer> ranks);

  void execute_kernel_unary(std::string func, id<MTLBuffer> A,
                            id<MTLBuffer> result, id<MTLBuffer> meta);

  void execute_kernel_init(std::string func, id<MTLBuffer> A,
                           id<MTLBuffer> meta);

  void initiate_dispatch_broadcastable(std::string kernel_method,
                                       const Tensor &a, const Tensor &b,
                                       Tensor &result);
  std::vector<id<MTLBuffer>> __dummy_data();
  void print_buffer_contents(std::vector<id<MTLBuffer>> buffers,
                             std::vector<int> stride);

  id<MTLBuffer> createBuffer(void *data, size_t size, DType type);
  id<MTLBuffer> createEmptyBuffer(int size, DType type);
  id<MTLBuffer> clone(id<MTLBuffer> buffer);
  void copy_vector_to_buffer(void *ptr, Memory &memory, int buffer_size);

  // arithmetic kernels
  void add(const Tensor &a, const Tensor &b, Tensor &result) override;
  void sub(const Tensor &a, const Tensor &b, Tensor &result) override;
  void mul(const Tensor &a, const Tensor &b, Tensor &result) override;
  void div(const Tensor &a, const Tensor &b, Tensor &result) override;

  // init kernels
  void ones(Tensor &a);
  void zeros(Tensor &a);

  // not implemented
  void eye(int n, DType dtype = DType::float32);
  void empty(std::vector<int> shape, DType dtype = DType::float32);
  void matmul(const Tensor &a, const Tensor &b, Tensor &result) override;
  void pow(const Tensor &a, const Tensor &b, Tensor &result) override;
};
