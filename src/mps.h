#ifndef MPS_H
#define MPS_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include <Metal/Metal.h>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
class MPS {
private:
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> commandQueue;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

public:
  MPS();
  void _init_pipeline(std::string metal_function_name);
  void execute_kernel_binary(std::string func, id<MTLBuffer> A, id<MTLBuffer> B,
                             id<MTLBuffer> result, id<MTLBuffer> meta);
  void execute_kernel_unary(std::string func, id<MTLBuffer> A,
                            id<MTLBuffer> result, id<MTLBuffer> meta);

  void execute_kernel_init(std::string func, id<MTLBuffer> A,
                           id<MTLBuffer> meta);

  std::vector<id<MTLBuffer>> __dummy_data();
  void print_buffer_contents(std::vector<id<MTLBuffer>> buffers,
                             std::vector<int> stride);
  template <typename Type> id<MTLBuffer> createBuffer(Type *data, size_t size);
  template <typename T> id<MTLBuffer> createEmptyBuffer(int size);
  id<MTLBuffer> clone(id<MTLBuffer> buffer);
};

#endif
