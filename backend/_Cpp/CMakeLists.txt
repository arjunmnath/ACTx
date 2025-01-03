cmake_minimum_required(VERSION 3.14)
project(MetalGraphs LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES "arm64")


set(METAL_SOURCES src/Shaders.metal)
set(METAL_LIB_NAME shader.metallib)

add_custom_target(compile_metal ALL
    COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCES} -o ${CMAKE_BINARY_DIR}/src/Shader.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/src/Shader.air -o ${CMAKE_BINARY_DIR}/src/${METAL_LIB_NAME}
    DEPENDS ${METAL_SOURCES}
    COMMENT "Compiling Metal shaders"
)
add_executable(out
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.mm"
)

add_custom_target(
    generate_shader_source
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/shader.mm"
)

add_dependencies(out generate_shader_source)


find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)

target_link_libraries(out
    PRIVATE
    "${FOUNDATION_FRAMEWORK}"
    "${METAL_FRAMEWORK}"
)

target_compile_options(out PRIVATE
    -fobjc-arc   
    -ObjC++      
    -g           
)

