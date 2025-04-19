#include "mps.h"
#include "device_type.h"
#include "main.h"
#include "types.h"
#include "utility.h"
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

static NSString *getModuleDirectory() {
  Dl_info info;
  if (dladdr((void *)&getModuleDirectory, &info) != 0 && info.dli_fname) {
    NSString *modulePath = [NSString stringWithUTF8String:info.dli_fname];
    NSString *moduleDir = [modulePath stringByDeletingLastPathComponent];
    return moduleDir;
  }
  return nil;
}
MPS::MPS() {
  NSError *error = nil;
  this->device = MTLCreateSystemDefaultDevice();
  if (!this->device) {
    std::cerr << "Metal not available" << std::endl;
    exit(1);
  }
  NSString *moduleDir = getModuleDirectory();
  NSString *metallibPath =
      [moduleDir stringByAppendingPathComponent:@"kernels.metallib"];
  NSData *libraryData = [NSData dataWithContentsOfFile:metallibPath];
  if (!libraryData) {
    NSLog(@"Error: Failed to load .metallib file");
  }
  dispatch_data_t dispatchData = dispatch_data_create(
      [libraryData bytes], [libraryData length], NULL, NULL);
  this->library = [device newLibraryWithData:dispatchData error:&error];
  if (!this->library) {
    std::cerr << "Shaders compilation failed "
              << [[error localizedDescription] UTF8String];
    exit(1);
  }
  this->commandQueue = [device newCommandQueue];
  if (!this->commandQueue) {
    std::cerr << "command queue creation failed" << std::endl;
    exit(1);
  }
}

void MPS::_init_pipeline(std::string metal_function_name) {
  NSError *error = nil;
  id<MTLFunction> function = [this->library
      newFunctionWithName:[NSString stringWithUTF8String:metal_function_name
                                                             .c_str()]];
  if (!function) {
    std::cerr << metal_function_name << " not found..." << std::endl;
    exit(1);
  }
  id<MTLComputePipelineState> pipelineState =
      [this->device newComputePipelineStateWithFunction:function error:&error];
  if (!pipelineState) {
    std::cerr << "Failed to create compute pipeline state for "
              << metal_function_name << " : "
              << [[error localizedDescription] UTF8String] << std::endl;
  }
  pipelines[metal_function_name] = pipelineState;
}
void MPS::execute_kernel_unary(std::string func, id<MTLBuffer> A,
                               id<MTLBuffer> result, id<MTLBuffer> meta) {
  std::string metal_function_name = func;
  if (!pipelines[metal_function_name]) {
    this->_init_pipeline(metal_function_name);
  }
  id<MTLComputePipelineState> pipelineState = pipelines[metal_function_name];
  id<MTLCommandBuffer> commandBuffer = [this->commandQueue commandBuffer];
  if (!commandBuffer) {
    std::cerr << "Failed to create command buffer." << std::endl;
    exit(1);
  }

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  if (!computeEncoder) {
    std::cerr << "Failed to create compute command encoder." << std::endl;
    exit(1);
  }
  [computeEncoder setComputePipelineState:pipelineState];
  [computeEncoder setBuffer:A offset:0 atIndex:0];
  [computeEncoder setBuffer:result offset:0 atIndex:1];
  NSUInteger threadExecutionWidth = pipelineState.threadExecutionWidth;
  size_t threadsPerThreadgroup = threadExecutionWidth;
  size_t threadgroups =
      (A.length + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::execute_kernel_init(std::string func, id<MTLBuffer> A,
                              id<MTLBuffer> meta) {
  std::string metal_function_name = func;
  if (!pipelines[metal_function_name]) {
    this->_init_pipeline(metal_function_name);
  }
  id<MTLComputePipelineState> pipelineState = pipelines[metal_function_name];
  id<MTLCommandBuffer> commandBuffer = [this->commandQueue commandBuffer];
  if (!commandBuffer) {
    std::cerr << "Failed to create command buffer." << std::endl;
    exit(1);
  }

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  if (!computeEncoder) {
    std::cerr << "Failed to create compute command encoder." << std::endl;
    exit(1);
  }
  [computeEncoder setComputePipelineState:pipelineState];
  [computeEncoder setBuffer:A offset:0 atIndex:0];
  [computeEncoder setBuffer:meta offset:0 atIndex:1];
  NSUInteger threadExecutionWidth = pipelineState.threadExecutionWidth;
  size_t threadsPerThreadgroup = threadExecutionWidth;
  size_t threadgroups =
      (A.length + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::execute_kernel_binary(std::string func, id<MTLBuffer> A,
                                id<MTLBuffer> B, id<MTLBuffer> result,
                                id<MTLBuffer> meta) {
  std::string metal_function_name = func;
  if (!pipelines[metal_function_name]) {
    this->_init_pipeline(metal_function_name);
  }
  id<MTLComputePipelineState> pipelineState = pipelines[metal_function_name];

  id<MTLCommandBuffer> commandBuffer = [this->commandQueue commandBuffer];
  if (!commandBuffer) {
    std::cerr << "Failed to create command buffer." << std::endl;
    exit(1);
  }

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  if (!computeEncoder) {
    std::cerr << "Failed to create compute command encoder." << std::endl;
    exit(1);
  }

  [computeEncoder setComputePipelineState:pipelineState];
  [computeEncoder setBuffer:A offset:0 atIndex:0];
  [computeEncoder setBuffer:B offset:0 atIndex:1];
  [computeEncoder setBuffer:result offset:0 atIndex:2];
  [computeEncoder setBuffer:meta offset:0 atIndex:3];
  NSUInteger threadExecutionWidth = pipelineState.threadExecutionWidth;
  size_t threadsPerThreadgroup = threadExecutionWidth;
  size_t threadgroups =
      (std::min(A.length, B.length) + threadsPerThreadgroup - 1) /
      threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

void MPS::execute_kernel_binary_with_broadcast(
    std::string func, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> result,
    id<MTLBuffer> lshape, id<MTLBuffer> rshape, id<MTLBuffer> target,
    id<MTLBuffer> ranks) {
  std::string metal_function_name = func;
  if (!pipelines[metal_function_name]) {
    this->_init_pipeline(metal_function_name);
  }
  id<MTLComputePipelineState> pipelineState = pipelines[metal_function_name];

  id<MTLCommandBuffer> commandBuffer = [this->commandQueue commandBuffer];
  if (!commandBuffer) {
    std::cerr << "Failed to create command buffer." << std::endl;
    exit(1);
  }

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  if (!computeEncoder) {
    std::cerr << "Failed to create compute command encoder." << std::endl;
    exit(1);
  }

  [computeEncoder setComputePipelineState:pipelineState];
  [computeEncoder setBuffer:A offset:0 atIndex:0];
  [computeEncoder setBuffer:B offset:0 atIndex:1];
  [computeEncoder setBuffer:result offset:0 atIndex:2];
  [computeEncoder setBuffer:lshape offset:0 atIndex:3];
  [computeEncoder setBuffer:rshape offset:0 atIndex:4];
  [computeEncoder setBuffer:target offset:0 atIndex:5];
  [computeEncoder setBuffer:ranks offset:0 atIndex:6];

  NSUInteger threadExecutionWidth = pipelineState.threadExecutionWidth;
  size_t threadsPerThreadgroup = threadExecutionWidth;
  size_t threadgroups =
      (std::min(A.length, B.length) + threadsPerThreadgroup - 1) /
      threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

id<MTLBuffer> MPS::createBuffer(void *data, size_t size, DType type) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithBytes:data
                                length:getDTypeSize(type) * size
                               options:MTLResourceStorageModeShared];
  return buffer;
}
id<MTLBuffer> MPS::createEmptyBuffer(int size, DType type) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithLength:getDTypeSize(type) * size
                                options:MTLResourceStorageModeShared];
  return buffer;
}

id<MTLBuffer> MPS::clone(id<MTLBuffer> buffer) {
  NSUInteger bufferSize = buffer.length;
  id<MTLBuffer> newBuffer =
      [device newBufferWithLength:bufferSize
                          options:MTLResourceStorageModeShared];
  void *originalData = buffer.contents;
  void *newData = newBuffer.contents;
  memcpy(newData, originalData, bufferSize);
  return newBuffer;
}

void MPS::copy_vector_to_buffer(void *ptr, Memory &memory, int buffer_size) {
  memcpy([memory.storage->metal contents], ptr, buffer_size);
}

void MPS::initiate_dispatch_broadcastable(std::string kernel_method,
                                          const Tensor &a, const Tensor &b,
                                          Tensor &result) {

  if (a.device != DeviceType::MPS || b.device != DeviceType::MPS ||
      result.device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  auto result_shape = compute_broadcast_shape(a, b);
  std::shared_ptr<Memory> lshape =
      pool->request_memory(DeviceType::MPS, a.dims.size(), DType::int32);
  this->copy_vector_to_buffer((void *)a.dims.data(), *lshape,
                              a.dims.size() * getDTypeSize(DType::int32));

  std::shared_ptr<Memory> rshape =
      pool->request_memory(DeviceType::MPS, b.dims.size(), DType::int32);
  this->copy_vector_to_buffer((void *)b.dims.data(), *rshape,
                              b.dims.size() * getDTypeSize(b.dtype));

  std::shared_ptr<Memory> target =
      pool->request_memory(DeviceType::MPS, result_shape.size(), a.dtype);
  this->copy_vector_to_buffer((void *)result_shape.data(), *target,
                              result_shape.size() * getDTypeSize(a.dtype));

  std::vector<int> _ranks = {static_cast<int>(a.dims.size()),
                             static_cast<int>(b.dims.size()),
                             static_cast<int>(result_shape.size())};

  std::shared_ptr<Memory> ranks =
      pool->request_memory(DeviceType::MPS, _ranks.size(), DType::int32);
  this->copy_vector_to_buffer((void *)_ranks.data(), *ranks,
                              result_shape.size() * getDTypeSize(DType::int32));

  this->execute_kernel_binary_with_broadcast(
      kernel_method, a.memory->storage->metal, b.memory->storage->metal,
      result.memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(&lshape->storage->metal),
      *reinterpret_cast<id<MTLBuffer> __strong *>(&rshape->storage->metal),
      *reinterpret_cast<id<MTLBuffer> __strong *>(&target->storage->metal),
      *reinterpret_cast<id<MTLBuffer> __strong *>(&ranks->storage->metal));
  pool->return_memory(lshape);
  pool->return_memory(rshape);
  pool->return_memory(target);
  pool->return_memory(ranks);
}

// ==================================================
//                     ARITHMETIC
// ==================================================
void MPS::add(const Tensor &a, const Tensor &b, Tensor &result) {
  this->initiate_dispatch_broadcastable("__add__", a, b, result);
};

void MPS::sub(const Tensor &a, const Tensor &b, Tensor &result) {
  this->initiate_dispatch_broadcastable("__sub__", a, b, result);
};
void MPS::mul(const Tensor &a, const Tensor &b, Tensor &result) {
  this->initiate_dispatch_broadcastable("__mul__", a, b, result);
};
void MPS::div(const Tensor &a, const Tensor &b, Tensor &result) {
  this->initiate_dispatch_broadcastable("__div__", a, b, result);
};

void MPS::matmul(const Tensor &a, const Tensor &b, Tensor &result) {
  throw std::logic_error("not implemented");
  this->initiate_dispatch_broadcastable("__matmul__", a, b, result);
};
void MPS::pow(const Tensor &a, const Tensor &b, Tensor &result) {
  // this->initiate_dispatch("__pow__", a, b, result);
}

// ==================================================
//                      INIT
// ==================================================

void MPS::ones(Tensor &a) {
  std::shared_ptr<Memory> meta_memory =
      pool->request_memory(DeviceType::MPS, 1, DType::int32);
  std::vector<int> meta = {
      std::accumulate(a.dims.begin(), a.dims.end(), 1, std::multiplies<int>())};
  this->copy_vector_to_buffer((void *)meta.data(), *meta_memory,
                              meta.size() * getDTypeSize(DType::int32));
  this->execute_kernel_init("__ones__", a.memory->storage->metal,
                            meta_memory->storage->metal);
  pool->return_memory(meta_memory);
}

void MPS::zeros(Tensor &a) {
  std::shared_ptr<Memory> meta_memory =
      pool->request_memory(DeviceType::MPS, 1, DType::int32);
  std::vector<int> meta = {
      std::accumulate(a.dims.begin(), a.dims.end(), 1, std::multiplies<int>())};
  this->copy_vector_to_buffer((void *)meta.data(), *meta_memory,
                              meta.size() * getDTypeSize(DType::int32));
  this->execute_kernel_init("__ones__", a.memory->storage->metal,
                            meta_memory->storage->metal);
  pool->return_memory(meta_memory);
}
