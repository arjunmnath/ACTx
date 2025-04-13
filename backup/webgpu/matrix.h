// matrix.h
#pragma once
#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>
#include <vector>
#include <webgpu/webgpu.h>

// Global WebGPU device reference
extern WGPUDevice g_device;
extern WGPUQueue g_queue;

// Function to initialize WebGPU from C++
bool initializeWebGPU();

class Matrix {
private:
  std::vector<float> data;
  size_t rows;
  size_t cols;

  // WebGPU resources
  WGPUBuffer bufferA = nullptr;
  WGPUBuffer bufferB = nullptr;
  WGPUBuffer bufferC = nullptr;
  WGPUShaderModule computeShaderModule = nullptr;
  WGPUComputePipeline computePipeline = nullptr;
  WGPUBindGroup bindGroup = nullptr;
  WGPUBindGroupLayout bindGroupLayout = nullptr;

  // Initialize GPU resources for this matrix
  void initGPUResources();
  void cleanupGPUResources();

public:
  Matrix(size_t rows, size_t cols);
  Matrix(const std::vector<std::vector<float>> &input);

  // Getters
  size_t getRows() const { return rows; }
  size_t getCols() const { return cols; }
  const std::vector<float> &getData() const { return data; }

  // Matrix operations
  Matrix add(const Matrix &other) const;

  // CPU fallback for matrix operations
  Matrix addCPU(const Matrix &other) const;

  // Utilities
  void print() const;
  std::vector<std::vector<float>> toNestedVector() const;

  // Destructor
  ~Matrix();
};

// JavaScript bindings
EMSCRIPTEN_BINDINGS(matrix_module) {
  emscripten::class_<Matrix>("Matrix")
      .constructor<size_t, size_t>()
      .constructor<const std::vector<std::vector<float>> &>()
      .function("add", &Matrix::add)
      .function("addCPU", &Matrix::addCPU)
      .function("getRows", &Matrix::getRows)
      .function("getCols", &Matrix::getCols)
      .function("getData", &Matrix::getData)
      .function("toNestedVector", &Matrix::toNestedVector)
      .function("print", &Matrix::print);

  emscripten::register_vector<float>("VectorFloat");
  emscripten::register_vector<std::vector<float>>("VectorVectorFloat");
  emscripten::function("initializeWebGPU", &initializeWebGPU);
}
