#include "memory_pool.h"
#include "memory.h"
#include "types.h"
#include <gtest/gtest.h>
#include <memory>


TEST(MemoryPoolTest, RequestAndReturnMemory) {
  MemoryPool pool;

  auto mem1 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  ASSERT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->size, 1024);
  EXPECT_EQ(mem1->dtype, DType::float32);

  pool.return_memory(mem1);

  auto mem2 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  EXPECT_EQ(mem2, mem1);
}

TEST(MemoryPoolTest, FindSuitableBlock) {
  MemoryPool pool;

  auto mem1 = pool.request_memory(DeviceType::MPS, 2048, DType::float32);
  pool.return_memory(mem1);

  auto found = pool.find_suitable_block(2048);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found, mem1);
}


TEST(MemoryPoolTest, MultipleBlocksAndReuse) {
  MemoryPool pool;

  auto m1 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  auto m2 = pool.request_memory(DeviceType::MPS, 2048, DType::float32);

  pool.return_memory(m1);
  pool.return_memory(m2);

  auto r1 = pool.request_memory(DeviceType::MPS, 1024, DType::float32);
  auto r2 = pool.request_memory(DeviceType::MPS, 2048, DType::float32);

  EXPECT_TRUE((r1 == m1 || r1 == m2));
  EXPECT_TRUE((r2 == m1 || r2 == m2));
  EXPECT_NE(r1, r2);
}


TEST(MemoryPoolTest, NoSuitableBlock) {
  MemoryPool pool;
  auto found = pool.find_suitable_block(4096);
  EXPECT_EQ(found, nullptr);
}


TEST(MemoryPoolTest, NonPowerOfTwoSizes) {
  MemoryPool pool;

  auto mem1 = pool.request_memory(DeviceType::MPS, 11, DType::float32);
  ASSERT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->size, 16);

  pool.return_memory(mem1);

  auto mem2 = pool.request_memory(DeviceType::MPS, 13, DType::float32);
  EXPECT_EQ(mem2, mem1);

  auto mem3 = pool.request_memory(DeviceType::MPS, 33, DType::float32);
  ASSERT_NE(mem3, nullptr);
  EXPECT_GE(mem3->size, 64);
  EXPECT_NE(mem3, mem1);

  pool.return_memory(mem2);
  pool.return_memory(mem3);
}
