#include "mps_helper.h"
#include <iostream>

MPSHelper::MPSHelper() { std::cout << "init cpu helper" << "\n"; }
void MPSHelper::print_msg() { std::cout << "running cpu print\n"; }
template <typename T> void MPSHelper::add(T *A, T *B) {
  std::cout << "mps add method" << std::endl;
}
