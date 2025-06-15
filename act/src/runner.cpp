#include "tensor.h"
#include <Foundation/Foundation.h>
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  long n = 4;
  Tensor *a = Tensor::eye(n, DType::float32, false);
  a->print();
  return 0;
}
