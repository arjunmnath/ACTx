#include "mps.h"
#include "device_type.h"
#include "main.h"
#include "types.h"
#import <objc/runtime.h>

#include "utility.h"
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <any>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <memory>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T>
void print_buffer(id<MTLBuffer> buffer, DType dtype, const char *label = "") {
  size_t count = buffer.length / getDTypeSize(dtype);
  std::vector<T> data(count);
  std::cout << count << std::endl;
  memcpy(data->data(), buffer.contents, buffer.length);
  std::cout << label << "[ ";
  for (size_t i = 0; i < count; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << "]" << std::endl;
}

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
  if (!libraryData)
    NSLog(@"Error: Failed to load .metallib file");
  dispatch_data_t dispatchData = dispatch_data_create(
      [libraryData bytes], [libraryData length], NULL, NULL);
  this->library = [device newLibraryWithData:dispatchData error:&error];
  if (!this->library) {
    std::cerr << "Shaders compilation failed "
              << [[error localizedDescription] UTF8String];
    exit(1);
  }
  this->commandQueue = [device newCommandQueue];
  MTLHeapDescriptor *heapDesc = [[MTLHeapDescriptor alloc] init];
  heapDesc.storageMode =
      MTLStorageModeShared; // or .Private for GPU-only access
  heapDesc.size =
      4096 * 10; // total heap size in bytes (align to resource size)
  heapDesc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
  heapDesc.type = MTLHeapTypeAutomatic;
  this->heap = [device newHeapWithDescriptor:heapDesc];
  if (!this->commandQueue) {
    std::cerr << "command queue creation failed" << std::endl;
    exit(1);
  }
}

std::pair<size_t, size_t> MPS::compute_threads(size_t N, size_t maxTPG) {
  uint32_t TPG = (N < maxTPG) ? N : maxTPG;
  uint32_t TGs = static_cast<uint32_t>(std::ceil(static_cast<float>(N) / TPG));
  return {TPG, TGs};
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
                               id<MTLBuffer> result, id<MTLBuffer> meta,
                               int N) {
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

  std::pair<size_t, size_t> threadinfo =
      this->compute_threads(N, pipelineState.threadExecutionWidth);
  [computeEncoder dispatchThreadgroups:MTLSizeMake(threadinfo.second, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadinfo.first, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::execute_kernel_init(std::string func, id<MTLBuffer> A,
                              id<MTLBuffer> meta, int N) {
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

  std::pair<size_t, size_t> threadinfo =
      this->compute_threads(N, pipelineState.maxTotalThreadsPerThreadgroup);
  [computeEncoder dispatchThreadgroups:MTLSizeMake(threadinfo.second, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadinfo.first, 1, 1)];

  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::execute_kernel_binary(std::string func, id<MTLBuffer> A,
                                id<MTLBuffer> B, id<MTLBuffer> result,
                                id<MTLBuffer> meta, int N) {
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

  std::pair<uint32_t, uint32_t> threadinfo =
      this->compute_threads(N, pipelineState.maxTotalThreadsPerThreadgroup);
  [computeEncoder dispatchThreadgroups:MTLSizeMake(threadinfo.second, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadinfo.first, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

void MPS::execute_kernel_binary_with_broadcast(std::string func,
                                               id<MTLBuffer> A, id<MTLBuffer> B,
                                               id<MTLBuffer> result,
                                               id<MTLBuffer> metadata, int N) {
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
  [computeEncoder setBuffer:metadata offset:0 atIndex:3];

  size_t threadsPerThreadgroup = pipelineState.threadExecutionWidth;
  size_t threadgroups =
      ((N + threadsPerThreadgroup - 1) / threadsPerThreadgroup) *
      threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::createEmptyBuffer(int size, DType type, Storage *storage) {
  static int val = 0;
  if (size <= 0) {
    throw std::runtime_error("invalid buffer size");
  }
  storage->metal =
      [this->heap newBufferWithLength:size * getDTypeSize(type)
                              options:MTLResourceStorageModeShared];
  if (!storage->metal) {
    throw std::runtime_error("Failed to allocate MTLBuffer");
  }
  /* NSLog(@"Buffer %@: %p, contents: %p, heap used: %lu", storage->metal.label,
   */
  /*       storage->metal, storage->metal.contents, this->heap.usedSize); */
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
  assert(memory.does_live_on(DeviceType::MPS));
  memcpy([memory.storage->metal contents], ptr, buffer_size);
}

void MPS::initiate_dispatch_broadcastable(std::string kernel_method,
                                          const Tensor *a, const Tensor *b,
                                          Tensor *result) {

  if (a->device != DeviceType::MPS || b->device != DeviceType::MPS ||
      result->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  std::vector<int> _ranks = {
      (int)result->size, static_cast<int>(a->dims.size()),
      static_cast<int>(b->dims.size()), static_cast<int>(result->dims.size())};
  std::vector<int> meta_data;
  meta_data.reserve(a->dims.size() + b->dims.size() + result->dims.size() + 4);
  meta_data.insert(meta_data.end(), _ranks.begin(), _ranks.end());
  meta_data.insert(meta_data.end(), a->dims.begin(), a->dims.end());
  meta_data.insert(meta_data.end(), b->dims.begin(), b->dims.end());
  meta_data.insert(meta_data.end(), result->dims.begin(), result->dims.end());
  Memory *meta_data_memory =
      pool->request_memory(DeviceType::MPS, meta_data.size(), DType::int32);
  this->copy_vector_to_buffer((void *)meta_data.data(), *meta_data_memory,
                              meta_data.size() * getDTypeSize(DType::int32));
  /*
    print_buffer<float>(a->memory->storage->metal, a->dtype, "A: ");
    print_buffer<float>(b->memory->storage->metal, b->dtype, "B: ");
    print_buffer<int>(ashape->storage->metal, ashape->dtype, "ashape: ");
    print_buffer<int>(bshape->storage->metal, bshape->dtype, "bshape: ");
    print_buffer<int>(result_shape_memory->storage->metal,
                      result_shape_memory->dtype, "Result Shape: ");
    print_buffer<int>(ranks->storage->metal, ranks->dtype, "Ranks: ");
    */

  this->execute_kernel_binary_with_broadcast(
      kernel_method, a->memory->storage->metal, b->memory->storage->metal,
      result->memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(
          &meta_data_memory->storage->metal),
      result->size);
  // print_buffer<float>(result->memory->storage->metal, result->dtype, "Result:
  // ");

  pool->return_memory(meta_data_memory);
}

void MPS::initiate_dispatch_comparison(std::string kernel_method,
                                       const Tensor *a, const Tensor *b,
                                       Tensor *result) {

  if (a->device != DeviceType::MPS || b->device != DeviceType::MPS ||
      result->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  Memory *meta =
      pool->request_memory(DeviceType::MPS, a->dims.size(), DType::int32);
  this->copy_vector_to_buffer((void *)a->dims.data(), *meta,
                              a->dims.size() * getDTypeSize(DType::int32));

  this->execute_kernel_binary(
      kernel_method, a->memory->storage->metal, b->memory->storage->metal,
      result->memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(&meta->storage->metal),
      result->size);
  pool->return_memory(meta);
}

void MPS::initiate_dispatch_init(std::string kernel_method, Tensor *a) {
  Memory *meta_memory = pool->request_memory(DeviceType::MPS, 1, DType::int32);
  this->copy_vector_to_buffer((void *)a->dims.data(), *meta_memory,
                              a->dims.size() * getDTypeSize(DType::int32));
  this->execute_kernel_init(kernel_method, a->memory->storage->metal,
                            meta_memory->storage->metal, a->size);
  pool->return_memory(meta_memory);
}

// ==================================================
//                     ARITHMETIC
//
// ==================================================
void MPS::negate(Tensor *a) { this->initiate_dispatch_init("__neg__", a); }
void MPS::add(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_broadcastable("__add__", a, b, result);
};

void MPS::sub(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_broadcastable("__sub__", a, b, result);
};
void MPS::mul(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_broadcastable("__mul__", a, b, result);
};
void MPS::div(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_broadcastable("__div__", a, b, result);
};

void MPS::matmul(const Tensor *a, const Tensor *b, Tensor *result) {
  throw std::logic_error("not implemented");
  this->initiate_dispatch_broadcastable("__matmul__", a, b, result);
};
void MPS::pow(const Tensor *a, const Tensor *b, Tensor *result) {
  if (a->device != DeviceType::MPS || b->device != DeviceType::MPS ||
      result->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  Memory *meta =
      pool->request_memory(DeviceType::MPS, a->dims.size(), DType::int32);
  this->copy_vector_to_buffer((void *)a->dims.data(), *meta,
                              a->dims.size() * getDTypeSize(DType::int32));
  execute_kernel_binary(
      "__pow__", a->memory->storage->metal, b->memory->storage->metal,
      result->memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(&meta->storage->metal),
      result->size);
  pool->return_memory(meta);
}
// ==================================================
//                      INIT
// ==================================================

void MPS::ones(Tensor *a) { this->initiate_dispatch_init("__ones__", a); }

void MPS::zeros(Tensor *a) { this->initiate_dispatch_init("__zeros__", a); }

void MPS::eye(Tensor *a) { this->initiate_dispatch_init("__eye__", a); }
void MPS::full(Tensor *n, Tensor *result) {
  if (n->device != DeviceType::MPS || result->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  Memory *meta =
      pool->request_memory(DeviceType::MPS, result->dims.size(), DType::int32);
  this->copy_vector_to_buffer((void *)result->dims.data(), *meta,
                              result->dims.size() * getDTypeSize(DType::int32));
  execute_kernel_unary(
      "__full__", result->memory->storage->metal, n->memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(&meta->storage->metal),
      result->size);
  pool->return_memory(meta);
}

// ==================================================
//                     COMPARISON
// ==================================================

void MPS::logical_e(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_e", a, b, result);
};

void MPS::logical_ne(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_ne", a, b, result);
};
void MPS::logical_gt(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_gt", a, b, result);
};

void MPS::logical_lt(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_lt", a, b, result);
};

void MPS::logical_gte(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_gte", a, b, result);
};

void MPS::logical_lte(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_comparison("logical_lte", a, b, result);
};
