#pragma once

#include "op_register.h"
#include "op_types.h"
#include <vector>
struct OpNode {
  Operation op;
  OPType type;
  std::vector<OpNode *> inputs;
  std::vector<OpNode *> outputs;
};
