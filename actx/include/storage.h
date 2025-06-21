#pragma once
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

struct Storage {
#ifdef __OBJC__
  __strong id<MTLBuffer> metal;
#endif
  void *cpu;

  Storage() : cpu(nullptr) {
#ifdef __OBJC__
    metal = nil; // Use nil instead of nullptr for Objective-C objects
#endif
  }

  void clear() {
#ifdef __OBJC__
    if (metal) {
      // If not using ARC, you might need: [metal release];
      metal = nil; // This should properly release under ARC
    }
#endif
    if (cpu) {
      // Don't set cpu to nullptr unless you've freed the memory
      // free(cpu); // if you allocated it
      cpu = nullptr;
    }
  }

  ~Storage() {
#ifdef __OBJC__
    if (metal) {
      NSLog(@"Releasing buffer: %@", metal.label);
      metal = nil; // Ensure the buffer is released
    }
#endif
    if (cpu) {
      // Clean up CPU memory if needed
      cpu = nullptr;
    }
  }
};
