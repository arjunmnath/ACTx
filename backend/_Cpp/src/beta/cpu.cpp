#include "device.h"
#include <iostream>

class CPUHelper : public Device {
public:
  CPUHelper() { std::cout << "init cpu helper" << "\n"; }
  template <typename T> void add(T *A, T *B) {
    std::cout << "cpu add method" << std::endl;
  }
  void print_msg() { std::cout << "running cpu print\n"; }
};
