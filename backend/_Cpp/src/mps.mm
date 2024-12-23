#include "mps.h"
#include "shader.mm"
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
using namespace std;
MPS::MPS() {
  NSError *error = nil;
  this->device = MTLCreateSystemDefaultDevice();
  if (!this->device) {
    cerr << "Metal not available" << endl;
    exit(1);
  }
  /*
   NSString *shaderPath = [[NSBundle mainBundle] pathForResource:@"Shaders"
                                                          ofType:@"metal"
                                                     inDirectory:@"src"];
   // NSString *shaderPath =
   // @"/Users/arjunmnath/dev/mlp-from-scratch/backend/_Cpp/src/Shaders.metal";
   NSString *shaderSource =
       [NSString stringWithContentsOfFile:shaderPath
                                 encoding:NSUTF8StringEncoding
                                    error:&error];

   if (!shaderSource) {
     cerr << "Loading shader source failed at path " << shaderPath << ", "
          << [[error localizedDescription] UTF8String] << endl;
     exit(1);
   }
   */
  this->library = [this->device newLibraryWithSource:shaderSource
                                             options:nil
                                               error:&error];
  if (!this->library) {
    cerr << "Shaders compilation failed"
         << [[error localizedDescription] UTF8String];
    exit(1);
  }
  this->commandQueue = [device newCommandQueue];
  if (!this->commandQueue) {
    cerr << "command queue creation failed" << endl;
    exit(1);
  }
}

void MPS::_init_pipeline(string metal_function_name) {
  NSError *error = nil;
  id<MTLFunction> function = [this->library
      newFunctionWithName:[NSString stringWithUTF8String:metal_function_name
                                                             .c_str()]];
  if (!function) {
    cerr << metal_function_name << " not found..." << endl;
    exit(1);
  }
  id<MTLComputePipelineState> pipelineState =
      [this->device newComputePipelineStateWithFunction:function error:&error];
  if (!pipelineState) {
    cerr << "Failed to create compute pipeline state for" << metal_function_name
         << " : " << [[error localizedDescription] UTF8String] << endl;
  }
  pipelines[metal_function_name] = pipelineState;
}
void MPS::add_matrix(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> result,
                     id<MTLBuffer> meta) {
  // TODO: fix this size requirement;
  const size_t size = 25;
  string metal_function_name = "add_matrix";
  if (!pipelines[metal_function_name]) {
    this->_init_pipeline(metal_function_name);
  }
  id<MTLComputePipelineState> pipelineState = pipelines[metal_function_name];

  id<MTLCommandBuffer> commandBuffer = [this->commandQueue commandBuffer];
  if (!commandBuffer) {
    cerr << "Failed to create command buffer." << endl;
    exit(1);
  }

  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];
  if (!computeEncoder) {
    cerr << "Failed to create compute command encoder." << endl;
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
      (size + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  [computeEncoder
       dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
  [computeEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}
vector<id<MTLBuffer>> MPS::__dummy_data() {
  const size_t size = 25;
  vector<float> a(size);
  vector<float> b(size);
  vector<float> result(size, 0.0f);

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
  vector<id<MTLBuffer>> buffers;
  // 8. Create Metal buffers
  id<MTLBuffer> bufferA =
      [device newBufferWithBytes:a.data()
                          length:sizeof(float) * size
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferB =
      [device newBufferWithBytes:b.data()
                          length:sizeof(float) * size
                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferResult =
      [device newBufferWithBytes:result.data()
                          length:sizeof(float) * size
                         options:MTLResourceStorageModeShared];

  vector<uint> constants = {M, N, P};
  id<MTLBuffer> constantsBuffer =
      [device newBufferWithBytes:constants.data()
                          length:sizeof(constants)
                         options:MTLResourceStorageModeShared];

  buffers.push_back(bufferA);
  buffers.push_back(bufferB);
  buffers.push_back(bufferResult);
  buffers.push_back(constantsBuffer);
  return buffers;
}

void MPS::print_buffer_contents(vector<id<MTLBuffer>> buffers, uint stride[]) {
  float *a = (float *)[buffers[0] contents];
  float *b = (float *)[buffers[1] contents];
  float *gpuResult = (float *)[buffers[2] contents];
  uint *meta = (uint *)[buffers[3] contents];

  uint M = meta[0];
  uint N = meta[1];
  uint P = meta[2];
  for (uint i = 0; i < M; i++) {
    for (uint j = 0; j < N; j++) {
      cout << setw(2) << setfill('0') << a[i * stride[0] + j * stride[1]]
           << " ";
    }
    cout << "\t";
    for (uint j = 0; j < N; j++) {
      cout << setw(2) << setfill('0') << b[i * stride[0] + j * stride[1]]
           << " ";
    }
    cout << "\t";
    for (uint j = 0; j < N; j++) {
      cout << setw(2) << setfill('0')
           << gpuResult[i * stride[0] + j * stride[1]] << " ";
    }
    cout << "\n";
  }
  cout << endl;
}
