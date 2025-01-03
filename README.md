# Metal Accelerated Tensor Library with Compute Graphs

A high-performance tensor library that leverages Metal for GPU acceleration, implements dynamic compute graphs for automatic differentiation (autograd), and provides Python bindings for ease of use in machine learning and scientific computing. 

**Note**: This project is primarily designed for learning and educational purposes. While it is capable of performing basic tensor operations and gradient computation, it is **not intended for production-level model building**. However, with further development, it could be used for simple model building tasks.

## Features

- **GPU Acceleration**: Utilizes Metal for efficient tensor computation on macOS devices.
- **Dynamic Compute Graphs**: Implements dynamic computation graphs for automatic differentiation, similar to autograd, enabling gradient computation for machine learning tasks.
- **Python Bindings**: Provides Python bindings for seamless integration with Python-based workflows.
- **High Performance**: Optimized for both CPU and GPU execution, ensuring maximum performance across platforms.
- **Educational Focus**: Aimed at helping users understand the underlying concepts of tensor operations, autograd, and GPU acceleration.

## Requirements

- macOS 10.15+ or iOS 13+ with Metal support
- Xcode 12+ with Command Line Tools
- Python 3.x (for Python bindings)
- CMake 3.x or higher (for building the project)

## Installation

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/metal-tensor-library.git
   cd metal-tensor-library
   ```

2. Build the C++/Objective-C++ library:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Install Python bindings:
   ```bash
   pip install .
   ```

### Pre-built Binaries

Coming soon...

## Usage

### C++/Objective-C++ API

```cpp
#include "MetalTensor.h"

int main() {
    MetalTensor tensor1 = MetalTensor::random({3, 3});
    MetalTensor tensor2 = MetalTensor::random({3, 3});
    
    // Define a simple computation
    MetalTensor result = tensor1 * tensor2;
    
    // Compute gradients
    result.backward();
    
    // Access the gradients
    MetalTensor grad = tensor1.grad();
    grad.print();
    return 0;
}
```

### Python API

```python
import metal_tensor as mt

# Create tensors
tensor1 = mt.random((3, 3), requires_grad=True)
tensor2 = mt.random((3, 3), requires_grad=True)

# Define a simple computation
result = tensor1 * tensor2

# Compute gradients
result.backward()

# Access the gradients
grad_tensor1 = tensor1.grad
grad_tensor2 = tensor2.grad

print("Gradient of tensor1:\n", grad_tensor1)
print("Gradient of tensor2:\n", grad_tensor2)
```

## Documentation

<!-- For detailed documentation on the API and advanced usage, refer to the [docs](docs). -->
No documentation understand it yourself 🤷🏻‍♂️


## Contributing

We welcome contributions to this project! To contribute, please fork the repository and submit a pull request with your changes. Make sure to follow the code style and add tests for new features or bug fixes.

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Run tests to verify your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Metal framework for GPU acceleration
- Python bindings via [pybind11](https://pybind11.readthedocs.io/)
- Inspired by various tensor libraries such as NumPy and TensorFlow, and automatic differentiation systems like autograd.
