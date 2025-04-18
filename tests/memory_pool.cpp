
#include "memory_pool.h"
#include "memory.h"
#include "types.h"
#include <cassert>
void test_request_and_return_memory() {
  MemoryPool pool;

  std::shared_ptr<Memory> mem1 =
      pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  assert(mem1 != nullptr);
  assert(mem1->size == 1024);
  assert(mem1->dtype == DType::float32);

  pool.return_memory(mem1);

  std::shared_ptr<Memory> mem2 =
      pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  assert(mem2 == mem1);
}

void test_find_suitable_block() {
  MemoryPool pool;

  std::shared_ptr<Memory> mem1 =
      pool.request_memory(DeviceType::MPS, 2048, DType::float32);
  pool.return_memory(mem1);

  std::shared_ptr<Memory> found = pool.find_suitable_block(2048);
  assert(found != nullptr);
  assert(found == mem1);
}

void test_multiple_blocks_and_reuse() {
  MemoryPool pool;

  auto m1 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  auto m2 = pool.request_memory(DeviceType::MPS, 2048, DType::float32);

  pool.return_memory(m1);
  pool.return_memory(m2);

  auto r1 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  auto r2 = pool.request_memory(DeviceType::MPS, 2048, DType::float32);

  assert((r1 == m1 || r1 == m2));
  assert((r2 == m1 || r2 == m2));
  assert(r1 != r2);
}

void test_no_suitable_block() {
  MemoryPool pool;
  std::shared_ptr<Memory> found = pool.find_suitable_block(4096);
  assert(found == nullptr);
}

void test_non_power_of_two_sizes() {
  MemoryPool pool;

  auto mem1 = pool.request_memory(DeviceType::MPS, 11, DType::float32);
  assert(mem1 != nullptr);
  assert(mem1->size >= 11);

  pool.return_memory(mem1);

  auto mem2 = pool.request_memory(DeviceType::MPS, 13, DType::float32);
  assert(mem2 == mem1);

  auto mem3 = pool.request_memory(DeviceType::MPS, 33, DType::float32);
  assert(mem3 != nullptr);
  assert(mem3->size >= 33);
  assert(mem3 != mem1);
  pool.return_memory(mem2);
  pool.return_memory(mem3);
}

int main() {
  test_request_and_return_memory();
  test_find_suitable_block();
  test_no_suitable_block();
  test_multiple_blocks_and_reuse();
  return 0;
}
