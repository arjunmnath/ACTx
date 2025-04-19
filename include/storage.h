#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

union Storage {
#ifdef __OBJC__
  id<MTLBuffer> metal;
#endif
  void *cpu;
  Storage() {};
  ~Storage() {};
  void clear() {
#ifdef __OBJC__
    if (metal) {
      metal = nullptr;
    }

#endif
    if (cpu) {
      cpu = nullptr;
    }
  }
};
