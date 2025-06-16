# ACTx (Accelerated, Computegraph, Tensor, Execution)

![PyPI version](https://img.shields.io/pypi/v/actx)
![PyPI downloads](https://img.shields.io/pypi/dm/actx)
![Build](https://github.com/arjunmnath/ACTx/actions/workflows/publish.yaml/badge.svg)
![Codecov](https://codecov.io/gh/arjunmnath/ACTx/branch/main/graph/badge.svg)
![License](https://img.shields.io/github/license/arjunmnath/ACTx)
<!--![Platform](https://img.shields.io/badge/platform-linux%20|%20macOS%20|%20windows-blue)-->
![Platform](https://img.shields.io/badge/platform-%20macOS%20-blue)
![Wheel](https://img.shields.io/pypi/wheel/actx)
![code style](https://img.shields.io/badge/code%20style-black-black)
![Python version](https://img.shields.io/pypi/pyversions/actx)

> "What I cannot create, I do not understand." - Richard Feynman

Driven by this philosophy, this high-performance tensor library was built as a hands-on learning project. It leverages Metal for GPU acceleration, implements dynamic compute graphs for automatic differentiation (autograd), and provides Python bindings for ease of use in machine learning and scientific computing.

**Note**: While capable of basic tensor operations and gradient computation, this project is intended primarily for educational purposes and is **not intended for production-level model building**.

## Features

- **GPU Acceleration**: Utilizes Metal for efficient tensor computation on macOS devices.
- **Dynamic Compute Graphs**: Implements dynamic computation graphs for automatic differentiation, similar to autograd, enabling gradient computation for machine learning tasks.
- **Python Bindings**: Provides Python bindings for seamless integration with Python-based workflows.
- **High Performance**: Optimized for both CPU and GPU execution, ensuring maximum performance across Metal enabled devices.
- **Educational Focus**: Aimed at helping users understand the underlying concepts of tensor operations, autograd, and GPU acceleration.

## Requirements

- macOS 10.15+ or iOS 13+ with Metal support
- Xcode 12+ with Command Line Tools
- Python 3.x (for Python bindings)
- CMake 3.x or higher (for building the project)
<!--

## Project Structure

````
.
├── src
│   └── beta
│   │   ├── cpu.cpp
│   │   ├── cpu.h
│   │   ├── device.cpp
│   │   ├── device.h
│   │   ├── mps_helper.mm
│   │   ├── mps_helper.h
│   │   └── tensor.mm
│   ├──  matrix.cpp
│   ├──  mps.h
│   ├──  mps.nm
│   ├── Shaders.metal
│   ├── tensor.mm
│   └── wrapper.cpp
├── tests
│   ├── CMakeLists.txt
│   ├── ...
│   └──
├── examples
│   └── mlp
│       ├── activations
│       │   ├── __init__.py
│       │   └── main.py
│       ├── costs
│       │   ├── __init__.py
│       │   └── main.py
│       ├── layers
│       │   ├── __init__.py
│       │   └── main.py
│       ├── optimizers
│       │   ├── __init__.py
│       │   └── main.py
│       ├── tensors
│       │   ├── __init__.py
│       │   └── tensor.py
│       ├── tests
│       │   ├── activation_methods.py
│       │   ├── cost_methods.py
│       │   └── layer.py
│       └── tf_impl
│           ├── data.json
│           ├── gpt-version.py
│           ├── mnist_model.h5
│           └── requirements.txt
├── .gitignore
├── build_ext.py
├── CMakeLists.txt
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── setup.py
└── setup.cfg
``` -->
## Installation

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/arjunmnath/ACTx.git
   cd ACTx
   ```

2. Build the C++/Objective-C++ library:

   > Debug build

   ```bash
   cmake --preset debug
   cmake --build build -- -j$(nproc)
   ```

   > Release build

   ```bash
   cmake --preset release
   cmake --build build -- -j$(nproc)
   ```

   > Test build

   ```bash
   cmake --preset test
   cmake --build build -- -j$(nproc)
   ```

   > Run Tests

   ```
   ctest --parallel $(nproc) --progress --test-dir build --output-on-failure
   ```

3. Install Python bindings:
   ```bash
   pip install .
   ```

### From PyPi

```bash
pip install actx
```

> 🚧 Only on MacOS at present

## Usage

### C++/Objective-C++ API

```cpp
#include "actx.h"

int main() {
    Tensor tensor1 = Tensor::random({3, 3});
    Tensor tensor2 = Tensor::random({3, 3});

    // Define a simple computation
    Tensor result = tensor1 * tensor2;

    // Compute gradients
    result.backward();

    // Access the gradients
    Tensor grad = tensor1.grad();
    grad.print();
    return 0;
}
```

### Python API

```python
import actx

# Create tensors
tensor1 = actx.random((3, 3), requires_grad=True)
tensor2 = actx.random((3, 3), requires_grad=True)

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

Contributions are welcome! Please read our [Contributions Guide](CONTRIBUTING.md) before submitting a pull request. If you encounter any issues, feel free to open an issue in the repository.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Metal framework for GPU acceleration
- Python bindings via [c-api](https://docs.python.org/3/c-api/)
- Inspired by various tensor libraries such as NumPy and PyTorch, and automatic differentiation systems like autograd.
