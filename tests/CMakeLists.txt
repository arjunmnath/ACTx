# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Set project name
project(MathOperationsTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the math_operations.cpp source files
add_library(math_operations STATIC op.cpp)

# Add the test source file
add_executable(tests op_test.cpp)

# Find GoogleTest package
find_package(GTest REQUIRED)

# Link GoogleTest and pthread (needed for GoogleTest)
target_link_libraries(tests math_operations GTest::GTest GTest::Main pthread)

# Enable testing
enable_testing()

# Add test to CTest (CMake's testing tool)
add_test(NAME MathOperationsTest COMMAND tests)
