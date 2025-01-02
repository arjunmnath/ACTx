#include <iostream>
#include <stdexcept>
#include <vector>

class Matrix {
private:
  std::vector<std::vector<double>> data; // Stores matrix elements
  size_t rows;                           // Number of rows
  size_t cols;                           // Number of columns

public:
  // Constructors
  Matrix(size_t rows, size_t cols) {
    // Body to initialize with dimensions
  }
  Matrix(const std::vector<std::vector<double>> &values) {
    // Body to initialize with a 2D vector
  }

  // Destructor
  ~Matrix() {
    // Body for destructor
  }

  // Accessors
  size_t getRows() const {
    // Body to return number of rows
  }
  size_t getCols() const {
    // Body to return number of columns
  }
  double getElement(size_t row, size_t col) const {
    // Body to get an element
  }
  void setElement(size_t row, size_t col, double value) {
    // Body to set an element
  }

  // Operators
  Matrix operator+(const Matrix &other) const {
    // Body for addition
  }
  Matrix operator-(const Matrix &other) const {
    // Body for subtraction
  }
  Matrix elementwiseMultiply(const Matrix &other) const {
    // Body for element-wise multiplication
  }
  Matrix matrixMultiply(const Matrix &other) const {
    // Body for matrix multiplication
  }
  Matrix operator*(double scalar) const {
    // Body for scalar multiplication
  }
  Matrix &operator=(const Matrix &other) {
    // Body for assignment
  }

  // Comparison operators
  bool operator==(const Matrix &other) const {
    // Body for equality check
  }
  bool operator!=(const Matrix &other) const {
    // Body for inequality check
  }

  // Utility methods
  Matrix transpose() const {
    // Body for transpose
  }
  Matrix inverse() const {
    // Body for inverse (square matrices only)
  }
  double determinant() const {
    // Body for determinant (square matrices only)
  }

  // Input/Output
  void print() const {
    // Body to print matrix
  }
};
