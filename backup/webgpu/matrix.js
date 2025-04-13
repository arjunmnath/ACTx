// matrix.js - JavaScript wrapper for WebAssembly matrix module

export class MatrixWrapper {
  constructor(input) {
    // Make sure Module is initialized
    if (!window.Module) {
      throw new Error("WebAssembly Module not initialized");
    }

    if (Array.isArray(input)) {
      // If input is a 2D array, create matrix from it
      console.log(input);
      this._matrix = new Module.Matrix([
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
      ]);
    } else if (typeof input === "object" && input._matrix) {
      // If input is another MatrixWrapper, copy its internal matrix
      this._matrix = input._matrix;
    } else {
      throw new Error("Invalid matrix initialization");
    }
  }

  // Matrix addition
  add(other) {
    if (!(other instanceof MatrixWrapper)) {
      throw new Error("Argument must be a Matrix");
    }

    const resultMatrix = this._matrix.add(other._matrix);
    return new MatrixWrapper(resultMatrix);
  }

  // Get matrix dimensions
  get rows() {
    return this._matrix.getRows();
  }

  get cols() {
    return this._matrix.getCols();
  }

  // Convert to JavaScript array
  toArray() {
    return this._matrix.toNestedVector();
  }

  // Print matrix to console
  print() {
    this._matrix.print();
  }

  // Clean up when done
  destroy() {
    if (this._matrix) {
      this._matrix.delete();
      this._matrix = null;
    }
  }
}

// Factory function for convenient matrix creation
export function Matrix(data) {
  return new MatrixWrapper(data);
}
