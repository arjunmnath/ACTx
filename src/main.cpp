#include "main.h"
#include "memory_pool.h"
#include "mps.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <memory>

std::unique_ptr<MemoryPool> pool = std::make_unique<MemoryPool>();
std::unique_ptr<MPS> mps = std::make_unique<MPS>();
std::shared_ptr<spdlog::logger> logger =
    spdlog::basic_logger_mt("file_logger", "logs.txt");
std::unique_ptr<Dispatcher> dispatcher = std::make_unique<Dispatcher>();
namespace {
int _init() {
  dispatcher->init_register();
  logger->set_level(spdlog::level::off);
  return 0;
}
int intialized = _init();
} // namespace
//
