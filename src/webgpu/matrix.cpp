// matrix.cpp
#include "matrix.h"
#include <cstring>
#include <emscripten.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Global WebGPU device and queue
WGPUDevice g_device = nullptr;
WGPUQueue g_queue = nullptr;

// The shader code for matrix addition
const char *COMPUTE_SHADER = R"(
    @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
    @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
    @group(0) @binding(2) var<storage, read_write> matrixC: array<f32>;
    
    struct Dimensions {
        rows: u32,
        cols: u32,
    }
    
    @group(0) @binding(3) var<uniform> dim: Dimensions;
    
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        if (global_id.x >= dim.cols || global_id.y >= dim.rows) {
            return;
        }
        
        let index = global_id.y * dim.cols + global_id.x;
        matrixC[index] = matrixA[index] + matrixB[index];
    }
)";

// C++ function to initialize WebGPU - we'll use JS interop for this
// but keep the control logic in C++
bool initializeWebGPU() {
  // Call to JavaScript to initialize WebGPU
  EM_ASM({
    if (!navigator.gpu) {
      console.error("WebGPU is not supported in this browser");
      return;
    }

    // Request adapter
    navigator.gpu.requestAdapter()
        .then(function(adapter) {
          if (!adapter) {
            console.error("Couldn't request WebGPU adapter");
            return;
          }

          // Request device
          adapter.requestDevice().then(function(device) {
            if (!device) {
              console.error("Couldn't request WebGPU device");
              return;
            }

            // Store device and queue in JavaScript
            window.gpuDevice = device;
            window.gpuQueue = device.queue;

            // Initialize the C++ callback
            // We can't directly pass pointers, but we can signal success
            ccall('webGPUInitialized', 'void', [], []);

            console.log("WebGPU initialized successfully");
          });
        })
        .catch(function(error) {
          console.error("WebGPU initialization error:", error);
        });
  });

  return true;
}

// This function is called from JavaScript when WebGPU is initialized
extern "C" {
EMSCRIPTEN_KEEPALIVE
void webGPUInitialized() {
  std::cout << "WebGPU device initialized from JavaScript" << std::endl;
  // We can't directly set g_device here because we can't pass WebGPU objects
  // Instead, when we need to use WebGPU, we'll access it via JavaScript
}
}

// Matrix constructor with dimensions
Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0f) {
  initGPUResources();
}

// Matrix constructor from nested vectors
Matrix::Matrix(const std::vector<std::vector<float>> &input) {
  if (input.empty() || input[0].empty()) {
    throw std::invalid_argument("Input matrix cannot be empty");
  }

  rows = input.size();
  cols = input[0].size();
  data.resize(rows * cols);

  // Copy data
  for (size_t i = 0; i < rows; ++i) {
    if (input[i].size() != cols) {
      throw std::invalid_argument(
          "All rows must have the same number of columns");
    }
    for (size_t j = 0; j < cols; ++j) {
      data[i * cols + j] = input[i][j];
    }
  }

  initGPUResources();
}

// Initialize GPU resources for this matrix
void Matrix::initGPUResources() {
  // We can't directly access g_device here due to the binding limitations
  // Instead, we'll set up resources when needed during operations
}

// Clean up GPU resources
void Matrix::cleanupGPUResources() {
  // Clean up buffers and other resources
  // This will be done through JavaScript calls
}

// Matrix addition
Matrix Matrix::add(const Matrix &other) const {
  if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }

  // Check if WebGPU is available through JavaScript
  bool webgpuAvailable = EM_ASM_INT({
    return (typeof window.gpuDevice != = 'undefined' &&
                                         window.gpuDevice != = null)
               ? 1
               : 0;
  });

  if (!webgpuAvailable) {
    std::cout << "WebGPU not available, using CPU fallback" << std::endl;
    return addCPU(other);
  }

  // Create output matrix
  Matrix result(rows, cols);

  // Perform WebGPU computation via JavaScript
  const size_t dataSize = rows * cols * sizeof(float);

  EM_ASM(
      {
        try {
          const device = window.gpuDevice;
          const queue = window.gpuQueue;

          // Get pointers to the data
          const ptrA = $0;
          const ptrB = $1;
          const ptrResult = $2;
          const dataSize = $3;
          const rows = $4;
          const cols = $5;

          // Create buffers
          const bufferA = device.createBuffer({
            size : dataSize,
            usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });

          const bufferB = device.createBuffer({
            size : dataSize,
            usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });

          const bufferC = device.createBuffer({
            size : dataSize,
            usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });

          const uniformBuffer = device.createBuffer({
            size : 8, // 2 uint32 values
            usage : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });

          // Create staging buffer for result readback
          const readbackBuffer = device.createBuffer({
            size : dataSize,
            usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          });

          // Copy data to buffers
          const dataA = new Float32Array(HEAPF32.buffer, ptrA, rows * cols);
          const dataB = new Float32Array(HEAPF32.buffer, ptrB, rows * cols);

          queue.writeBuffer(bufferA, 0, dataA);
          queue.writeBuffer(bufferB, 0, dataB);

          // Write uniform data (dimensions)
          const uniformData = new Uint32Array([ rows, cols ]);
          queue.writeBuffer(uniformBuffer, 0, uniformData);

          // Create shader module
          const shaderModule = device.createShaderModule({
            code:
              ` @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
              @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
              @group(0) @binding(2) var<storage, read_write> matrixC
                  : array<f32>;

              struct Dimensions {
                rows : u32, cols : u32,
              }

              @group(0) @binding(3) var<uniform>
                  dim : Dimensions;

              @compute @workgroup_size(8, 8, 1) fn main(
                  @builtin(global_invocation_id) global_id : vec3<u32>) {
                if (global_id.x >= dim.cols || global_id.y >= dim.rows) {
                  return;
                }

                let index = global_id.y * dim.cols + global_id.x;
                matrixC[index] = matrixA[index] + matrixB[index];
              }
              `
          });

          // Create bind group layout and pipeline layout
          const bindGroupLayout = device.createBindGroupLayout({
            entries : [
              {
                binding : 0,
                visibility : GPUShaderStage.COMPUTE,
                buffer : {type : "read-only-storage"}
              },
              {
                binding : 1,
                visibility : GPUShaderStage.COMPUTE,
                buffer : {type : "read-only-storage"}
              },
              {
                binding : 2,
                visibility : GPUShaderStage.COMPUTE,
                buffer : {type : "storage"}
              },
              {
                binding : 3,
                visibility : GPUShaderStage.COMPUTE,
                buffer : {type : "uniform"}
              }
            ]
          });

          const pipelineLayout = device.createPipelineLayout(
              {bindGroupLayouts : [bindGroupLayout]});

          // Create compute pipeline
          const computePipeline = device.createComputePipeline({
            layout : pipelineLayout,
            compute : {module : shaderModule, entryPoint : "main"}
          });

          // Create bind group
          const bindGroup = device.createBindGroup({
            layout : bindGroupLayout,
            entries : [
              {binding : 0, resource : {buffer : bufferA}},
              {binding : 1, resource : {buffer : bufferB}},
              {binding : 2, resource : {buffer : bufferC}},
              {binding : 3, resource : {buffer : uniformBuffer}}
            ]
          });

          // Create command encoder
          const commandEncoder = device.createCommandEncoder();
          const computePass = commandEncoder.beginComputePass();
          computePass.setPipeline(computePipeline);
          computePass.setBindGroup(0, bindGroup);

          // Dispatch workgroups
          const workgroupSizeX = 8;
          const workgroupSizeY = 8;
          const workgroupCountX = Math.ceil(cols / workgroupSizeX);
          const workgroupCountY = Math.ceil(rows / workgroupSizeY);
          computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
          computePass.end();

          // Copy result to readback buffer
          commandEncoder.copyBufferToBuffer(bufferC, 0, readbackBuffer, 0,
                                            dataSize);

          // Submit commands
          const commands = commandEncoder.finish();
          queue.submit([commands]);

          // Map buffer to read the result
          readbackBuffer.mapAsync(GPUMapMode.READ).then(() = > {
            const resultData =
                new Float32Array(readbackBuffer.getMappedRange());

            // Copy result back to C++
            const resultHeap =
                new Float32Array(HEAPF32.buffer, ptrResult, rows * cols);
            resultHeap.set(resultData);

            // Unmap the buffer
            readbackBuffer.unmap();

            console.log("WebGPU matrix addition completed");
          });
        } catch (error) {
          console.error("WebGPU operation failed:", error);
          // Call the CPU fallback
          _matrix_addCPU_fallback($0, $1, $2, $4, $5);
        }
      },
      data.data(), other.data.data(), result.data.data(), dataSize, rows, cols);

  return result;
}

// This function is called from JavaScript if the GPU operation fails
extern "C" {
EMSCRIPTEN_KEEPALIVE
void matrix_addCPU_fallback(float *a, float *b, float *result, size_t rows,
                            size_t cols) {
  std::cout << "Using CPU fallback from JS callback" << std::endl;
  for (size_t i = 0; i < rows * cols; ++i) {
    result[i] = a[i] + b[i];
  }
}
}

// CPU implementation of matrix addition
Matrix Matrix::addCPU(const Matrix &other) const {
  if (rows != other.rows || cols != other.cols) {
    throw std::invalid_argument("Matrix dimensions must match for addition");
  }

  Matrix result(rows, cols);

  for (size_t i = 0; i < data.size(); ++i) {
    result.data[i] = data[i] + other.data[i];
  }

  return result;
}

// Print matrix (for debugging)
void Matrix::print() const {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << data[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

// Convert to nested vector (for JavaScript interop)
std::vector<std::vector<float>> Matrix::toNestedVector() const {
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result[i][j] = data[i * cols + j];
    }
  }
  return result;
}

// Destructor
Matrix::~Matrix() { cleanupGPUResources(); }
