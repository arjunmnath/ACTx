#ifndef DEVICE_H
#define DEVICE_H
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include <Metal/Metal.h>
class Device {
public:
  virtual void add();
  virtual ~Device() = default;
};

#endif
