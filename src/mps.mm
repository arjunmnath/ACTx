#include "mps.h"
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

template <typename Type>
id<MTLBuffer> MPS::createBuffer(Type *data, size_t size) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithBytes:data
                                length:sizeof(Type) * size
                               options:MTLResourceStorageModeShared];
  return buffer;
}
template <typename T> id<MTLBuffer> MPS::createEmptyBuffer(int size) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithLength:sizeof(T) * size
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

// type definitions
template id<MTLBuffer> MPS::createEmptyBuffer<int>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<float>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<uint8_t>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<int8_t>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<bool>(int size);

template id<MTLBuffer> MPS::createBuffer<int>(int *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<float>(float *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<uint8_t>(uint8_t *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<int8_t>(int8_t *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<bool>(bool *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<int const>(int const *data,
                                                    size_t size);
template id<MTLBuffer> MPS::createBuffer<const float>(const float *data,
                                                      size_t size);
template id<MTLBuffer> MPS::createBuffer<const uint8_t>(const uint8_t *data,
                                                        size_t size);
template id<MTLBuffer> MPS::createBuffer<const int8_t>(const int8_t *data,
                                                       size_t size);
template id<MTLBuffer>MPS::createBuffer<const bool>(const bool *data, size_t size);
