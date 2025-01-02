#ifndef CPU_H
#define CPU_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "device.h"
class CPUHelper : public Device {
public:
  CPUHelper();
  void print_msg();
  template <typename T> void add(T *A, T *B);
};
#endif
