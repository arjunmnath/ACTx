#include "types.h"
#include "utility.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

// Test case for DType::int8
void test_int8() {
  int result = getDTypeSize(DType::int8);
  assert(result == 1 && "Expected size 1 for int8");
}

// Test case for DType::float16
void test_float16() {
  int result = getDTypeSize(DType::float16);
  assert(result == 2 && "Expected size 2 for float16");
}

// Test case for DType::int16
void test_int16() {
  int result = getDTypeSize(DType::int16);
  assert(result == 2 && "Expected size 2 for int16");
}

// Test case for DType::float32
void test_float32() {
  int result = getDTypeSize(DType::float32);
  assert(result == 4 && "Expected size 4 for float32");
}

// Test case for DType::int32
void test_int32() {
  int result = getDTypeSize(DType::int32);
  assert(result == 4 && "Expected size 4 for int32");
}

// Test case for DType::float64
void test_float64() {
  int result = getDTypeSize(DType::float64);
  assert(result == 8 && "Expected size 8 for float64");
}

// Test case for DType::int64
void test_int64() {
  int result = getDTypeSize(DType::int64);
  assert(result == 8 && "Expected size 8 for int64");
}

// Test case for invalid DType
void test_invalid_type() {
  try {
    int result = getDTypeSize(static_cast<DType>(999));
    assert(false && "Division by invalid DType should throw an exception");
  } catch (const std::invalid_argument &e) {
    assert(true && "Caught expected exception for invalid DType");
  }
}

int main() {
  test_int8();
  test_float16();
  test_int16();
  test_float32();
  test_int32();
  test_float64();
  test_int64();
  test_invalid_type();
  std::cout << "All tests passed!" << std::endl;

  return 0;
}
