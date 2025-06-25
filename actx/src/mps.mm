#include "mps.h"
#include "device_type.h"
#include "main.h"
#include "types.h"
#include "utility.h"
#include <Foundation/Foundation.h>
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
#include <objc/runtime.h>
#include <optional>
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
  memcpy(data.data(), buffer.contents, buffer.length);
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

void MPS::execute_kernel_nullary(std::string func, id<MTLBuffer> A,
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
void MPS::execute_kernel_unary(std::string func, id<MTLBuffer> input,
                               id<MTLBuffer> output, id<MTLBuffer> metadata,
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
  [computeEncoder setBuffer:input offset:0 atIndex:0];
  [computeEncoder setBuffer:output offset:0 atIndex:1];
  [computeEncoder setBuffer:metadata offset:0 atIndex:2];
  std::pair<size_t, size_t> threadinfo =
      this->compute_threads(N, pipelineState.threadExecutionWidth);
  [computeEncoder dispatchThreadgroups:MTLSizeMake(threadinfo.second, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threadinfo.first, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
void MPS::execute_kernel_binary(std::string func, id<MTLBuffer> A,
                                id<MTLBuffer> B, id<MTLBuffer> result,
                                id<MTLBuffer> meta, int N, int offset_a,
                                int offset_b, int offset_result) {
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
  [computeEncoder setBuffer:A offset:offset_a atIndex:0];
  [computeEncoder setBuffer:B offset:offset_b atIndex:1];
  [computeEncoder setBuffer:result offset:offset_result atIndex:2];
  [computeEncoder setBuffer:meta offset:0 atIndex:3];

  std::pair<uint32_t, uint32_t> threadinfo =
      this->compute_threads(N, pipelineState.maxTotalThreadsPerThreadgroup);

  [computeEncoder dispatchThreadgroups:MTLSizeMake(threadinfo.second, 1, 1)

                 threadsPerThreadgroup:MTLSizeMake(threadinfo.first, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

void MPS::initiate_dispatch_nullary(std::string kernel_method, Tensor *input) {
  if (input->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  std::vector<int> _ranks = {static_cast<int>(input->size),
                             static_cast<int>(input->dims.size())};
  std::vector<int> meta_data;
  meta_data.reserve(input->dims.size() + 2);
  meta_data.insert(meta_data.end(), _ranks.begin(), _ranks.end());
  meta_data.insert(meta_data.end(), input->dims.begin(), input->dims.end());
  Memory *meta_data_memory =
      pool->request_memory(DeviceType::MPS, meta_data.size(), DType::int32);
  this->copy_vector_to_buffer((void *)meta_data.data(), *meta_data_memory,
                              meta_data.size() * getDTypeSize(DType::int32));
  this->execute_kernel_nullary(kernel_method, input->memory->storage->metal,
                               *reinterpret_cast<id<MTLBuffer> __strong *>(
                                   &meta_data_memory->storage->metal),
                               input->size);
  pool->return_memory(meta_data_memory);
};
void MPS::initiate_dispatch_unary(std::string kernel_method,
                                  const Tensor *input, Tensor *output) {
  if (input->device != DeviceType::MPS || output->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  std::vector<int> _ranks = {static_cast<int>(output->size),
                             static_cast<int>(input->dims.size()),
                             static_cast<int>(output->dims.size())};
  std::vector<int> meta_data;
  meta_data.reserve(input->dims.size() + output->dims.size() + 3);
  meta_data.insert(meta_data.end(), _ranks.begin(), _ranks.end());
  meta_data.insert(meta_data.end(), input->dims.begin(), input->dims.end());
  meta_data.insert(meta_data.end(), output->dims.begin(), output->dims.end());
  Memory *meta_data_memory =
      pool->request_memory(DeviceType::MPS, meta_data.size(), DType::int32);
  this->copy_vector_to_buffer((void *)meta_data.data(), *meta_data_memory,
                              meta_data.size() * getDTypeSize(DType::int32));

  this->execute_kernel_unary(kernel_method, input->memory->storage->metal,
                             output->memory->storage->metal,
                             *reinterpret_cast<id<MTLBuffer> __strong *>(
                                 &meta_data_memory->storage->metal),
                             output->size);
  pool->return_memory(meta_data_memory);
};
void MPS::initiate_dispatch_binary(std::string kernel_method, const Tensor *a,
                                   const Tensor *b, Tensor *result) {

  if (a->device != DeviceType::MPS || b->device != DeviceType::MPS ||
      result->device != DeviceType::MPS) {
    throw std::runtime_error("All the tensor must live in Metal Buffers");
  }
  std::vector<int> _ranks = {
      (int)result->size, static_cast<int>(a->dims.size()),
      static_cast<int>(b->dims.size()), static_cast<int>(result->dims.size())};
  std::vector<int> meta_data;
  meta_data.reserve(
      (a->dims.size() + b->dims.size() + result->dims.size()) * 2 + 4);
  meta_data.insert(meta_data.end(), _ranks.begin(), _ranks.end());
  meta_data.insert(meta_data.end(), a->dims.begin(), a->dims.end());
  meta_data.insert(meta_data.end(), a->stride.begin(), a->stride.end());
  meta_data.insert(meta_data.end(), b->dims.begin(), b->dims.end());
  meta_data.insert(meta_data.end(), b->stride.begin(), b->stride.end());
  meta_data.insert(meta_data.end(), result->dims.begin(), result->dims.end());
  meta_data.insert(meta_data.end(), result->stride.begin(),
                   result->stride.end());
  Memory *meta_data_memory =
      pool->request_memory(DeviceType::MPS, meta_data.size(), DType::int32);
  this->copy_vector_to_buffer((void *)meta_data.data(), *meta_data_memory,
                              meta_data.size() * getDTypeSize(DType::int32));
  this->execute_kernel_binary(
      kernel_method, a->memory->storage->metal, b->memory->storage->metal,
      result->memory->storage->metal,
      *reinterpret_cast<id<MTLBuffer> __strong *>(
          &meta_data_memory->storage->metal),
      result->size, a->offset() * getDTypeSize(a->dtype),
      b->offset() * getDTypeSize(b->dtype),
      result->offset() * getDTypeSize(result->dtype));
  pool->return_memory(meta_data_memory);
};

void MPS::createEmptyBuffer(int bytesize, DType type, Storage *storage) {
  if (bytesize <= 0) {
    throw std::runtime_error("invalid buffer size");
  }
  storage->metal =
      [this->heap newBufferWithLength:bytesize
                              options:MTLResourceStorageModeShared];

  /* NSLog(@"Buffer: %@", storage->metal); */
  if (!storage->metal) {
    throw std::runtime_error("Failed to allocate MTLBuffer");
  }
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
// ==================================================
//                     ARITHMETIC
// ==================================================
void MPS::negate(Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__neg__", input, output);
}
void MPS::add(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("__add__", a, b, result);
};

void MPS::sub(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("__sub__", a, b, result);
};
void MPS::mul(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("__mul__", a, b, result);
};
void MPS::div(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("__div__", a, b, result);
};

void MPS::matmul(const Tensor *a, const Tensor *b, Tensor *result) {
  throw std::logic_error("not implemented");
  this->initiate_dispatch_binary("__matmul__", a, b, result);
};
void MPS::pow(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("__pow__", a, b, result);
}
// ==================================================
//                      INIT
// ==================================================
void MPS::ones(Tensor *a) { this->initiate_dispatch_nullary("__ones__", a); }

void MPS::zeros(Tensor *a) { this->initiate_dispatch_nullary("__zeros__", a); }

void MPS::eye(Tensor *a) { this->initiate_dispatch_nullary("__eye__", a); }

void MPS::full(Tensor *n, Tensor *result) {
  assert(n->size == 1);
  initiate_dispatch_unary("__full__", n, result);
}

// ==================================================
//                     COMPARISON
// ==================================================

void MPS::logical_e(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_e", a, b, result);
};

void MPS::logical_ne(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_ne", a, b, result);
};
void MPS::logical_gt(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_gt", a, b, result);
};

void MPS::logical_lt(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_lt", a, b, result);
};

void MPS::logical_gte(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_gte", a, b, result);
};

void MPS::logical_lte(const Tensor *a, const Tensor *b, Tensor *result) {
  this->initiate_dispatch_binary("logical_lte", a, b, result);
};

// ==================================================
//                    MATH FUNCTIONS
// ==================================================
void MPS::sqrt(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__sqrt__", input, output);
}
void MPS::exp(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__exp__", input, output);
}
void MPS::log(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__log__", input, output);
}
void MPS::log10(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__log10__", input, output);
}
void MPS::log2(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__log2__", input, output);
}
// ==================================================
//                    TRIG FUNCTIONS
// ==================================================
void MPS::sin(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__sin__", input, output);
}
void MPS::cos(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__cos__", input, output);
}

void MPS::tan(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__tan__", input, output);
}
void MPS::asin(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__asin__", input, output);
}
void MPS::acos(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__acos__", input, output);
}
void MPS::atan(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__atan__", input, output);
}
void MPS::atan2(const Tensor *x, const Tensor *y, Tensor *result) {
  this->initiate_dispatch_binary("__atan2__", x, y, result);
}
// ==================================================
//              HYPERBOLIC FUNCTIONS
// ==================================================
void MPS::sinh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__sinh__", input, output);
}
void MPS::cosh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__cosh__", input, output);
}

void MPS::tanh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__tanh__", input, output);
}
void MPS::asinh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__asinh__", input, output);
}
void MPS::acosh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__acosh__", input, output);
}
void MPS::atanh(const Tensor *input, Tensor *output) {
  this->initiate_dispatch_unary("__atanh__", input, output);
}

void MPS::rand(Tensor *a, Tensor *meta) {
  std::vector<type_t> data;
  /* id<MTLBuffer> meta = device_mps->ckkreateBuffer(shape.data(), 2, dtype); */
  /* int size = */
  /*     std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
   */
  /* std::vector<float> data(size, 0); */
  /* for (int i = 0; i < size; i++) { */
  /*   data[i] = __randn(); */
  /* } */
  /* id<MTLBuffer> result = device_mps->createBuffer(data.data(), size, dtype);
   */
  /* return Tensor(result, shape); */
  /* float __rand(int seed) { */
}
