#include "op.h"
#include <gtest/gtest.h>

// Test case for addition
TEST(MathOperationsTest, Addition) {
  EXPECT_EQ(add(2, 3), 5);    // Test with positive numbers
  EXPECT_EQ(add(-1, 1), 0);   // Test with negative and positive number
  EXPECT_EQ(add(-2, -3), -5); // Test with negative numbers
  EXPECT_EQ(add(0, 0), 0);    // Test with zeros
}

// Test case for subtraction
TEST(MathOperationsTest, Subtraction) {
  EXPECT_EQ(subtract(5, 3), 2);   // Test with positive numbers
  EXPECT_EQ(subtract(-1, 1), -2); // Test with negative and positive number
  EXPECT_EQ(subtract(-2, -3), 1); // Test with negative numbers
  EXPECT_EQ(subtract(0, 0), 0);   // Test with zeros
}
