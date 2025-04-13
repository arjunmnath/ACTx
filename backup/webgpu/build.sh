#!/bin/bash
# Build script for compiling the matrix library to WebAssembly

# Make sure Emscripten is in your PATH
# If you haven't installed it, follow instructions at https://emscripten.org/docs/getting_started/downloads.html

# Create build directory
mkdir -p build
cd build

# Compile C++ to WebAssembly with WebGPU support
emcc ../matrix.cpp \
  -o matrix.js \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ASSERTIONS=2 \
  -s USE_WEBGPU=1 \
  -s EXPORT_NAME="Module" \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s "EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']" \
  -s "EXPORTED_FUNCTIONS=['_malloc', '_free', '_webGPUInitialized', '_matrix_addCPU_fallback']" \
  -lembind \
  -std=c++17 \
  -O2

echo "Build complete. JavaScript module and WebAssembly binary are in the build directory."
