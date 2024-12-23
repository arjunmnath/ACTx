#import <Foundation/Foundation.h>

NSString *shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_matrix(device float *A [[buffer(0)]],      // Matrix A
                       device float *B [[buffer(1)]],      // Matrix B
                       device float *C [[buffer(2)]],      // Result matrix C
                       constant uint3 &dims [[buffer(3)]], // buffer for M, N, P
                       uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;    // Rows in A
  uint N = dims.y;    // Columns in A / Rows in B
  uint P = dims.z;    // Columns in B
  uint row = tid / P; // Compute row index for C
  uint col = tid % P; // Compute column index for C

  if (row < M && col < P) {
    C[row * P + col] = A[row * P + col] + B[row * P + col];
  }
}

kernel void subtract_matrix(device float *A [[buffer(0)]], // Matrix A
                            device float *B [[buffer(1)]], // Matrix B
                            device float *C [[buffer(2)]], // Result matrix C
                            constant uint3 &dims
                            [[buffer(3)]], // buffer for M, N, P
                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;    // Rows in A
  uint N = dims.y;    // Columns in A / Rows in B
  uint P = dims.z;    // Columns in B
  uint row = tid / P; // Compute row index for C
  uint col = tid % P; // Compute column index for C
  if (row < M && col < P) {
    C[row * P + col] = A[row * P + col] - B[row * P + col];
  }
}

kernel void matrix_multiply(device float *A [[buffer(0)]], // Matrix A
                            device float *B [[buffer(1)]], // Matrix B
                            device float *C [[buffer(2)]], // Result matrix C
                            constant uint3 &dims
                            [[buffer(3)]], // buffer for M, N, P

                            uint tid [[thread_position_in_grid]]) {

  uint M = dims.x;    // Rows in A
  uint N = dims.y;    // Columns in A / Rows in B
  uint P = dims.z;    // Columns in B
  uint row = tid / P; // Compute row index for C
  uint col = tid % P; // Compute column index for C

  if (row < M && col < P) {
    float sum = 0.0;
    for (uint k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * P + col];
    }
    C[row * P + col] = sum;
  }
}
)";
