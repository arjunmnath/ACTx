cmake_minimum_required(VERSION 3.14)
project(MetalGraphs LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/kernels")
set(METAL_BUILD_DIR "${CMAKE_BINARY_DIR}/build")
set(METAL_LIB_NAME "kernels.metallib")

file(GLOB METAL_SOURCES "${METAL_SOURCE_DIR}/*.metal")

file(MAKE_DIRECTORY ${METAL_BUILD_DIR})

foreach(METAL_FILE ${METAL_SOURCES})
    get_filename_component(METAL_FILENAME ${METAL_FILE} NAME_WE)
    set(AIR_FILE "${METAL_BUILD_DIR}/${METAL_FILENAME}.air")

    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal -c ${METAL_FILE} -o ${AIR_FILE}
        DEPENDS ${METAL_FILE}
        COMMENT "Compiling ${METAL_FILENAME}.metal to ${METAL_FILENAME}.air"
    )
    list(APPEND AIR_FILES ${AIR_FILE})
endforeach()

set(METALLIB_FILE "${CMAKE_BINARY_DIR}/${METAL_LIB_NAME}")
add_custom_command(
    OUTPUT ${METALLIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_FILE}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking .air files to ${METAL_LIB_NAME}"
)

add_custom_target(compile_metal ALL
    DEPENDS ${METALLIB_FILE}
    COMMENT "Building Metal library ${METAL_LIB_NAME}"
)

add_custom_command(
    TARGET compile_metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_FILE} ${METAL_BUILD_DIR}
    COMMENT "Copying ${METAL_LIB_NAME} to build directory"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build selected")
    set(SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/runner.cpp"
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Test")
    message(STATUS "Test build selected")
    set(SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.mm"

    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build selected")
    set(SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.mm"
    )
endif()

add_executable(out ${SOURCE_FILES})


find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT FOUNDATION_FRAMEWORK)
    message(FATAL_ERROR "Foundation framework not found!")
endif()

find_library(METAL_FRAMEWORK Metal)

target_link_libraries(out
    PRIVATE
    ${FOUNDATION_FRAMEWORK}   
    ${METAL_FRAMEWORK}        
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(out PRIVATE
        -DDEBUG_BUILD  
        -g           
        -O0          
        -ObjC++
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Test")
    target_compile_options(out PRIVATE
        -DTEST_BUILD  
        -g  
        -O2  
        -ObjC++
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(out PRIVATE
        -DRELEASE_BUILD  
        -O3              
        -DNDEBUG         
        -ObjC++
        -fsanitize=address
    )
endif()

target_compile_options(out PRIVATE
    -fobjc-arc   
)

target_include_directories(out PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")


