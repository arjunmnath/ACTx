#ifndef MPS_H
#define MPS_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include <Metal/Metal.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
class MPS {
private:
private:
  id<MTLDevice> device;
  id<MTLLibrary> library;
  id<MTLCommandQueue> commandQueue;
  unordered_map<string, id<MTLComputePipelineState>> pipelines;

public:
  MPS();
  void _init_pipeline(string metal_function_name);
  void add_matrix(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> result,
                  id<MTLBuffer> meta);
  vector<id<MTLBuffer>> __dummy_data();
  void print_buffer_contents(vector<id<MTLBuffer>> buffers, uint stride[]);
};

#endif
