#pragma once

#include "dispatcher.h"
#include "memory_pool.h"
#include "mps.h"

extern std::unique_ptr<MemoryPool> pool;
extern std::unique_ptr<Dispatcher> dispatcher;
extern std::unique_ptr<MPS> mps;
