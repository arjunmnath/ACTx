cmake_minimum_required(VERSION 3.14)
project(MetalGraphs LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")
cmake_policy(SET CMP0135 NEW)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.2
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(spdlog)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories(${gtest_SOURCE_DIR}/googletest/include)

# manually mention
file(GLOB_RECURSE OBJCXX_HEADERS "${CMAKE_SOURCE_DIR}/include/mps.h"
     "${CMAKE_SOURCE_DIR}/include/memory.h"
     "${CMAKE_SOURCE_DIR}/include/storage.h")
set_source_files_properties(${OBJCXX_HEADERS}
                            PROPERTIES COMPILE_FLAGS "-x objective-c++")

# Directories
set(METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/kernels")
set(METAL_BUILD_DIR "${CMAKE_BINARY_DIR}/build")
set(METAL_LIB_NAME "kernels.metallib")
set(TEST_DIR "${CMAKE_SOURCE_DIR}/tests")

# Files
file(GLOB METAL_SOURCES "${METAL_SOURCE_DIR}/*.metal")
file(GLOB TEST_SOURCES "${TEST_DIR}/*.cpp")
file(GLOB ALL_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_SOURCE_DIR}/src/*.mm")

# command source files
set(COMMAN_SOURCES ${ALL_SOURCES})
list(REMOVE_ITEM COMMAN_SOURCES "${CMAKE_SOURCE_DIR}/src/runner.cpp"
     "${CMAKE_SOURCE_DIR}/src/wrapper.cpp")
# DEBUG and RELEASE specific sources
set(DEBUG_SOURCES "")
foreach(source ${COMMAN_SOURCES})
  list(APPEND DEBUG_SOURCES ${source})
endforeach()
list(APPEND DEBUG_SOURCES "${CMAKE_SOURCE_DIR}/src/runner.cpp")

set(RELEASE_SOURCES "")
foreach(source ${COMMAN_SOURCES})
  list(APPEND RELEASE_SOURCES ${source})
endforeach()
list(APPEND RELEASE_SOURCES "${CMAKE_SOURCE_DIR}/src/wrapper.cpp")

message(STATUS "Filtered COMMAND SOURCES: ${COMMAN_SOURCES}")
message(STATUS "DEBUG specific sources: ${DEBUG_SOURCES}")
message(STATUS "RELEASE specific sources: ${RELEASE_SOURCES}")

# Ensure build directory exists
file(MAKE_DIRECTORY ${METAL_BUILD_DIR})

# Compile Metal shaders to .air files
set(AIR_FILES)
foreach(METAL_FILE ${METAL_SOURCES})
  get_filename_component(METAL_FILENAME ${METAL_FILE} NAME_WE)
  set(AIR_FILE "${METAL_BUILD_DIR}/${METAL_FILENAME}.air")

  add_custom_command(
    OUTPUT ${AIR_FILE}
    COMMAND xcrun -sdk macosx metal -c ${METAL_FILE} -o ${AIR_FILE}
    DEPENDS ${METAL_FILE}
    COMMENT "Compiling ${METAL_FILENAME}.metal to ${METAL_FILENAME}.air"
    VERBATIM)
  list(APPEND AIR_FILES ${AIR_FILE})
endforeach()

# Link .air files into a .metallib
set(METALLIB_FILE "${CMAKE_BINARY_DIR}/${METAL_LIB_NAME}")
add_custom_command(
  OUTPUT ${METALLIB_FILE}
  COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_FILE}
  DEPENDS ${AIR_FILES}
  COMMENT "Linking .air files to ${METAL_LIB_NAME}"
  VERBATIM)

# Custom target to build Metal library
add_custom_target(
  compile_metal ALL
  DEPENDS ${METALLIB_FILE}
  COMMENT "Building Metal library ${METAL_LIB_NAME}")

# Copy .metallib to build directory
add_custom_command(
  TARGET compile_metal
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_FILE} ${METAL_BUILD_DIR}
  COMMENT "Copying ${METAL_LIB_NAME} to build directory")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
if(NOT FOUNDATION_FRAMEWORK OR NOT METAL_FRAMEWORK)
  message(FATAL_ERROR "Required frameworks not found!")
endif()
set(COMMON_LIBRARIES ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} spdlog::spdlog)
# Build configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Debug build selected")
  add_executable(out ${DEBUG_SOURCES})
  target_compile_options(
    out
    PRIVATE -DDEBUG_BUILD -g -O0 -ObjC++ -fobjc-arc
            # -fsanitize=address
            -Wno-unused-command-line-argument)

  target_link_libraries(out PRIVATE ${COMMON_LIBRARIES}) # -fsanitize=address)
elseif(CMAKE_BUILD_TYPE STREQUAL "Test")
  enable_testing()
  set(SOURCE_FILES ${COMMAN_SOURCES} ${TEST_SOURCES})
  message(STATUS "Test build selected")
  message(STATUS "Test BUILD SOURCES: ${SOURCE_FILES}")
  add_executable(AllTests ${SOURCE_FILES})
  target_link_libraries(AllTests PRIVATE ${COMMON_LIBRARIES} gtest_main)
  target_compile_options(
    AllTests PRIVATE -DTEST_BUILD -O0 -g -ObjC++ # -fsanitize=address
                     -Wno-unused-command-line-argument)
  # target_link_libraries(AllTests PRIVATE -fsanitize=address)
  include(GoogleTest)
  gtest_discover_tests(AllTests)

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Release build selected")
  # TODO: ADD The python.h reference
  add_executable(out ${RELEASE_SOURCES})
  target_link_libraries(out PRIVATE ${COMMON_LIBRARIES})
  target_compile_options(out PRIVATE -DRELEASE_BUILD -O3 -DNDEBUG -ObjC++
                                     -fobjc-arc)
endif()
