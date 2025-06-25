#pragma once

#include "op_register.h"
#include "op_types.h"
#include "tensor.h"
#include <vector>
struct OpNode {
  Operation *op = nullptr;
  OPType type;
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
};
