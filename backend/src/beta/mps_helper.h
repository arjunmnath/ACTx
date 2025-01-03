#ifndef MPS_H
#define MPS_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device.h"
class MPSHelper : public Device {
public:
  MPSHelper();
  void print_msg();
  template <typename T> void add(T *A, T *B);
};
#endif
