#pragma once

#include "dispatcher.h"
#include "memory_pool.h"

extern std::unique_ptr<MemoryPool> pool;
extern std::unique_ptr<Dispatcher> dispatcher;
