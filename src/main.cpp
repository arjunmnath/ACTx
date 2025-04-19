#include "main.h"
#include "memory_pool.h"
#include "mps.h"
#include <memory>

std::unique_ptr<MemoryPool> pool = std::make_unique<MemoryPool>();
std::unique_ptr<MPS> mps = std::make_unique<MPS>();

std::unique_ptr<Dispatcher> dispatcher = std::make_unique<Dispatcher>();
namespace {
int _init_dispatcher() {
  dispatcher->init_register();
  return 0;
}
int dispatcher_initialized = _init_dispatcher();
} // namespace
