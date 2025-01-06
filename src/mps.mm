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
      [libraryData bytes], [libraryData length], dispatch_get_main_queue(),
      ^{
      });
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
  [computeEncoder setBuffer:meta offset:0 atIndex:2];
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

template <typename Type>
id<MTLBuffer> MPS::createBuffer(Type *data, size_t size) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithBytes:data
                                length:sizeof(Type) * size
                               options:MTLResourceStorageModeShared];
  return buffer;
}
template id<MTLBuffer> MPS::createBuffer<int>(int *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<float>(float *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<uint8_t>(uint8_t *data, size_t size);
template id<MTLBuffer> MPS::createBuffer<int8_t>(int8_t *data, size_t size);

template <typename T> id<MTLBuffer> MPS::createEmptyBuffer(int size) {
  id<MTLBuffer> buffer =
      [this->device newBufferWithLength:sizeof(T) * size
                                options:MTLResourceStorageModeShared];
  return buffer;
}

template id<MTLBuffer> MPS::createEmptyBuffer<int>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<float>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<uint8_t>(int size);
template id<MTLBuffer> MPS::createEmptyBuffer<int8_t>(int size);

std::vector<id<MTLBuffer>> MPS::__dummy_data() {
  const size_t size = 25;
  std::vector<float> a(size);
  std::vector<float> b(size);
  std::vector<float> result(size, 0.0f);

  uint32_t M = 5;
  uint32_t N = 5;
  uint32_t P = 5;

  // Initialize arrays with random data
  uint32_t k = size;
  for (size_t i = 0; i < size; ++i) {
    a[i] = static_cast<float>(k);
    b[i] = static_cast<float>(size - k);
    k--;
  }
  std::vector<id<MTLBuffer>> buffers;
  std::vector<uint> constants = {M, N, P};
  buffers.push_back(this->createBuffer(a.data(), size));
  buffers.push_back(this->createBuffer(b.data(), size));
  buffers.push_back(this->createBuffer(result.data(), size));
  buffers.push_back(this->createBuffer(constants.data(), constants.size()));

  return buffers;
}

void MPS::print_buffer_contents(std::vector<id<MTLBuffer>> buffers,
                                std::vector<int> stride) {
  float *a = (float *)[buffers[0] contents];
  float *b = (float *)[buffers[1] contents];
  float *gpuResult = (float *)[buffers[2] contents];
  uint *meta = (uint *)[buffers[3] contents];
  uint M = meta[0];
  uint N = meta[1];
  for (uint i = 0; i < M; i++) {
    for (uint j = 0; j < N; j++) {
      std::cout << std::setw(2) << std::setfill('0')
                << a[i * stride[0] + j * stride[1]] << " ";
    }
    std::cout << "\t";
    for (uint j = 0; j < N; j++) {
      std::cout << std::setw(2) << std::setfill('0')
                << b[i * stride[0] + j * stride[1]] << " ";
    }
    std::cout << "\t";
    for (uint j = 0; j < N; j++) {
      std::cout << std::setw(2) << std::setfill('0')
                << gpuResult[i * stride[0] + j * stride[1]] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}
