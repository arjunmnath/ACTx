cmake_minimum_required(VERSION 3.14)
project(MetalGraphs LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_OSX_ARCHITECTURES "arm64")


# Define the source directory for Metal shaders
set(METAL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/kernels")
set(METAL_BUILD_DIR "${CMAKE_BINARY_DIR}/build")
set(METAL_LIB_NAME "kernels.metallib")

# Gather all .metal files in the source directory
file(GLOB METAL_SOURCES "${METAL_SOURCE_DIR}/*.metal")

# Ensure the build directory exists
file(MAKE_DIRECTORY ${METAL_BUILD_DIR})


# Compile each .metal file into a .air file
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

# Combine all .air files into a .metallib file
set(METALLIB_FILE "${CMAKE_BINARY_DIR}/${METAL_LIB_NAME}")
add_custom_command(
    OUTPUT ${METALLIB_FILE}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_FILE}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking .air files to ${METAL_LIB_NAME}"
)


# Add a custom target to ensure the metallib is built
add_custom_target(compile_metal ALL
    DEPENDS ${METALLIB_FILE}
    COMMENT "Building Metal library ${METAL_LIB_NAME}"
)

# Optionally copy the .metallib file to the backend directory
add_custom_command(
    TARGET compile_metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_BUILD_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_FILE} ${METAL_BUILD_DIR}
    COMMENT "Copying ${METAL_LIB_NAME} to build directory"
)

# add_custom_target(compile_metal ALL
#     COMMAND xcrun -sdk macosx metal -c ${METAL_SOURCES} -o ${CMAKE_BINARY_DIR}/shader.air
#     COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/shader.air -o ${CMAKE_BINARY_DIR}/${METAL_LIB_NAME}
#     DEPENDS ${METAL_SOURCES}
#     COMMENT "Compiling Metal shaders"
# )

add_executable(out
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mps.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.mm"
)



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

