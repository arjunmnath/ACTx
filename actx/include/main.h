#pragma once

#include "dispatcher.h"
#include "memory_pool.h"
#include "mps.h"
#include <spdlog/spdlog.h>

extern std::unique_ptr<MemoryPool> pool;
extern std::unique_ptr<Dispatcher> dispatcher;
extern std::unique_ptr<MPS> mps;
extern std::shared_ptr<spdlog::logger> logger;
