# Class Diagram

![draftv1](https://github.com/arjunmnath/MetalGraphs/blob/main/design/draft.svg)



# Sync() 
`sync()` is a deceptively simple but crucial part of a device backend. It ensures that **asynchronous operations on a device complete before proceeding** — especially important for GPU backends like Metal and WebGPU.

Let’s walk through what `sync()` means and how to implement it per device:

---

## 🔁 `sync()` — Concept

```cpp
virtual void sync() = 0;
```

### ✅ When to call it:
- Before reading back from GPU to CPU
- Before measuring performance
- For deterministic behavior in testing/debugging

---

## 🧠 Per Device Backend

### 🖥️ **CPU**
CPU ops are synchronous (unless you're threading manually), so:

```cpp
void CPU::sync() override {
    // No-op
}
```

---

### 🧲 **MPS (Metal Performance Shaders)**

Metal is *asynchronous* by design. You submit a `MTLCommandBuffer` to the GPU queue — it may execute *later*.

```cpp
void MPS::sync() override {
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}
```

- `commit()` submits the buffer
- `waitUntilCompleted()` blocks the CPU until GPU finishes

If you reuse command buffers, ensure each op creates or reuses one safely.

---

### 🌐 **WebGPU**

WebGPU also runs operations async via command queues. You’d typically:

```cpp
void WebGPU::sync() override {
    queue.onSubmittedWorkDone().await();  // pseudo-code
}
```

In practice, you'd have to use:
- `queue.submit([commands])`
- Then use the JS `GPUQueue.onSubmittedWorkDone()` promise, wrapped in native async logic.

In native C++ or Rust via Dawn/WGPU:
```cpp
void WebGPU::sync() override {
    device->Poll(true); // blocks until GPU queue is idle
}
```

---

## 🧩 Where You Use `sync()`

### Example: Reading tensor from GPU to CPU
```cpp
Tensor t = tensor_on_gpu;
t.device()->sync();
void* cpu_data = t.to_cpu().data();
```

### Example: Testing
```cpp
add(t1, t2);
t1.device()->sync();
assertEqual(t1.data(), expected);
```

---

## 🧠 Bonus Tips

- Keep `sync()` lightweight when possible (no full flush unless needed)
- Use it inside `.to_cpu()` conversions
- Provide `device->async()` or `stream()` if supporting advanced execution control later

---


# Dispatcher Logic
this is an important decision, and both **registry-based dispatch** and **virtual-method-based dispatch (via abstract base class)** are valid. Let’s explore both in depth with pros, cons, and when to use each.

---

## 🧭 Option 1: **Virtual Method Dispatch (OOP-style)**

### ✅ How it works

Each device (like `CPU`, `MPS`, `WebGPU`) inherits from an abstract `Device` class and **overrides** methods like `add`, `sub`, etc.

```cpp
class Device {
public:
    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
};

class CPU : public Device {
public:
    Tensor add(const Tensor& a, const Tensor& b) override;
};
```

Then you call:
```cpp
a.device()->add(a, b);
```

---

### ✅ Pros

- **Elegant** and **polymorphic** — code is easy to reason about
- Automatically dispatches based on object type (`virtual` dispatch)
- Keeps device-specific logic isolated

### ❌ Cons

- **Rigid**: You must define all ops in the abstract base class
- Hard to **dynamically register** new ops or backends
- Adding an op means updating every subclass (unless you use `default` impls)

---

## 🧭 Option 2: **Op Registry Dispatch (Data-driven)**

### ✅ How it works

Operations are registered at runtime in a global registry, indexed by `(op_name, device_name)`.

```cpp
OpRegistry::register_op("add", "cpu", &CPU::add);
```

Then when you want to perform an operation:

```cpp
auto fn = OpRegistry::get("add", a.device()->name());
Tensor result = fn(a, b);
```

---

### ✅ Pros

- **Extensible**: Easy to register new ops/devices/plugins at runtime
- **Decoupled**: Logic is not tightly bound to class hierarchy
- Perfect for ops that are **independent functions**, like `sin`, `matmul`, `conv2d`

### ❌ Cons

- Slightly more **boilerplate**
- Can be harder to navigate: no clear class-based organization
- Need to manage registry lifetimes carefully

---

## 📌 Suggested Hybrid Architecture (Best of Both Worlds)

Many modern ML libraries (like PyTorch) use a **hybrid** approach:

- Devices implement core memory + execution APIs via abstract class:
  
  ```cpp
  class Device {
      virtual Memory<T> alloc(size_t size) = 0;
      virtual void sync() = 0;
      std::string name();  // "cpu", "mps", etc
  };
  ```

- All **ops** are dispatched through a global **OpRegistry**:
  
  ```cpp
  add(a, b) → OpRegistry["add"][a.device()->name()](a, b)
  ```

This separation lets you:
- Keep device management modular (via `Device`)
- Make op registration dynamic and data-driven

---

## 🧩 What I Recommend for You

Since you're building a **PyTorch-like system**:

| Concern                          | Recommendation      |
|----------------------------------|----------------------|
| Core memory management, sync     | ✅ Use abstract `Device` class |
| Operation dispatch               | ✅ Use `OpRegistry` |
| Extending ops/devices later      | ✅ Registry scales better |
| Want per-device custom logic     | 🔄 Use lambdas or bound member functions in registry |

---

### Bonus: What PyTorch Does

PyTorch uses something like:

- `TensorImpl` holds device + dispatch keys
- C++ `DispatchStub` uses hashmaps for kernel lookup
- Separate "device context" and "operation registry"

It's closer to a **registry-style system with polymorphic helpers**.

---

# Dispatcher <-> Tensor Logic
Absolutely — here’s a clear C++-like pseudocode snippet showing:

- How `Tensor::add()` calls `Dispatcher`
- How `Dispatcher` uses `OpRegistry` to get the right backend function

---

## 🧱 Step-by-Step Components

### 1. `Tensor` (User API layer)

```cpp
class Tensor {
public:
    Device* device;
    void* data;
    Shape shape;
    DType dtype;

    Tensor add(const Tensor& other) const {
        return Dispatcher::call("add", *this, other);
    }
};
```

---

### 2. `Dispatcher` (Central dispatch layer)

```cpp
class Dispatcher {
public:
    static Tensor call(const std::string& op, const Tensor& a, const Tensor& b) {
        std::string device_name = a.device->name();  // e.g., "cpu", "mps"
        auto op_fn = OpRegistry::get(op, device_name);
        return op_fn(a, b);  // Call the actual function
    }
};
```

---

### 3. `OpRegistry` (Global registry of all op-device implementations)

```cpp
using OpFunc = std::function<Tensor(const Tensor&, const Tensor&)>;

class OpRegistry {
    static inline std::unordered_map<std::string,
        std::unordered_map<std::string, OpFunc>> registry;

public:
    static void register_op(const std::string& op_name, const std::string& device_name, OpFunc fn) {
        registry[op_name][device_name] = fn;
    }

    static OpFunc get(const std::string& op_name, const std::string& device_name) {
        return registry.at(op_name).at(device_name);
    }
};
```

---

### 4. Registering device-specific ops

```cpp
namespace CPU {
    Tensor add(const Tensor& a, const Tensor& b) {
        // Actual CPU addition logic here
        return result;
    }
}

void init_ops() {
    OpRegistry::register_op("add", "cpu", &CPU::add);
    // Register more ops and devices similarly
}
```

---

### ✅ Usage

```cpp
init_ops();  // Register all ops at startup

Tensor A = ...;  // on CPU
Tensor B = ...;

Tensor C = A.add(B);  // Dispatches to CPU::add via Dispatcher → OpRegistry
```

# Future improvements

### 1. **Memory Management:**
   - **Memory Pooling:** As it stands, the design allocates memory directly for each tensor via `alloc()`. For performance, you might consider **memory pooling** (especially on the GPU). Repeated allocations and deallocations can be costly. A pool can manage a chunk of memory and reuse it, reducing overhead.
   - **Lazy Allocation:** The `Tensor<T>` class may allocate memory when it's first used. However, it could be more efficient to defer memory allocation until the tensor is explicitly initialized or the first operation is performed, especially for large tensors.

### 2. **Device Synchronization:**
   - **Multi-Device Synchronization:** Handling computations across multiple devices (e.g., using both CPU and GPU simultaneously) is crucial in modern deep learning frameworks. The design could include an abstraction layer for managing device synchronization, ensuring operations on tensors that span devices are synchronized properly. This could include barriers, events, or sync flags to manage dependencies between operations.
   - **Async Operations:** The framework could benefit from **asynchronous execution** of tensor operations (e.g., using CUDA streams, MPS queues, or WebGPU's async capabilities) to allow operations to run concurrently while minimizing idle times.

### 3. **Error Handling and Fault Tolerance:**
   - **Error Propagation:** The system could integrate more robust error handling, especially around device failures or unsupported operations. For instance, if a `Device` (like `MPS` or `WebGPU`) is unavailable, the system could automatically fall back to another device, or throw meaningful exceptions that help users debug issues quickly.
   - **Memory Overflows:** Currently, there’s no mention of protections against out-of-bounds access or memory overflows. Implementing boundary checks or assertion mechanisms when accessing tensor elements could prevent bugs and crashes in unsafe operations.

### 4. **Tensor Operations Optimization:**
   - **JIT Compilation:** The `OpRegister` class stores function pointers for operations, but this could be made more efficient by incorporating **Just-In-Time (JIT) compilation**. JIT could optimize tensor operations on the fly, generating device-specific code for each operation (e.g., optimized for different tensor shapes and hardware architectures), leading to more efficient code execution.
   - **Operator Fusion:** For performance, you could implement **operator fusion**, where multiple operations on tensors (like a sequence of additions and multiplications) are fused into a single kernel, reducing memory access overhead and improving computational efficiency.

### 5. **Tensor Slicing and Views:**
   - **Advanced Slicing:** The current design only mentions basic slicing with `view()`. You might want to extend slicing functionality to support more complex views, including **striding, broadcasting**, or **advanced indexing** (like boolean masks or fancy indexing), similar to what PyTorch and NumPy offer.
   - **Memory Views (Shared Memory):** The `view()` function likely creates a non-owning reference to a portion of the tensor. It would be beneficial to support **memory views** that can be shared between tensors without copying the data, avoiding unnecessary memory allocations.

### 6. **Thread Safety & Concurrency:**
   - **Thread-Safe Operations:** In multi-threaded environments, operations like `add()`, `mul()`, and `sub()` need to be thread-safe. The current design seems to rely on manual locking (`lock()` and `unlock()`), but a more fine-grained approach such as **atomic operations** or **thread-local storage** might be more efficient for certain use cases.
   - **Parallelism in Operations:** For large tensors, matrix operations could be parallelized across multiple threads or devices. For example, implementing **parallel reductions** for element-wise operations or matrix multiplication could significantly speed up the framework on multi-core CPUs or GPUs.

### 7. **Dispatcher Efficiency:**
   - **Operation Dispatching**: The `Dispatcher` currently uses the `OpRegister` to look up operations, which could introduce overhead. A potential improvement would be to cache the results of operation lookups to reduce the cost of finding the correct device-specific implementation. This could be achieved via a **cache** of function pointers or operation implementations.
   - **Template Specialization**: Instead of a generic `Func` type for operations, you might consider **template specialization** for different tensor data types and devices. This can help avoid unnecessary runtime dispatching and allow for compile-time optimizations.

### 8. **Support for Sparse Tensors:**
   - **Sparse Tensors:** Currently, the design doesn't mention sparse tensors, which are commonly used in deep learning for memory efficiency. Implementing a sparse tensor format (e.g., Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC)) would allow operations on large sparse matrices while consuming less memory and being more efficient.

### 9. **Cross-Device Operations:**
   - **Cross-Device Tensor Operations:** The design assumes that a tensor is tied to a single device, but in practice, tensors often need to be transferred between devices (e.g., from CPU to GPU). While the `toDevice()` function exists, the design could benefit from a **more seamless cross-device operation** system, where tensors automatically manage their location and transfer when needed.
   - **Unified Memory Management:** Instead of manually managing the device-specific memory, the framework could integrate **unified memory management** (like CUDA’s managed memory or Metal’s unified memory). This would allow the system to transparently handle memory transfers between devices and simplify code for the user.

### 10. **Compatibility with Existing Libraries:**
   - **Interoperability:** If the goal is to make this library compatible with existing frameworks (e.g., NumPy, TensorFlow, or PyTorch), it would be beneficial to allow **easy conversion** between tensor types (e.g., converting a NumPy array to a `Tensor<T>`). Implementing this would allow users to easily leverage the power of this framework without having to rewrite code that already uses other libraries.
   - **Serialization:** It would be helpful to add **serialization** support, allowing tensors to be saved to disk and loaded later. This is especially useful for training models and checkpoints.

### 11. **Documentation and User-Friendliness:**
   - **Higher-Level API:** While the design provides low-level control, a higher-level API for users who want to perform common tasks (e.g., matrix multiplication, element-wise operations) without manually managing device-specific code would make the framework more user-friendly.
   - **Better Documentation:** The use of complex data types (like `Func`, `OpRegister`, `Device`) should be well-documented with examples, as this could make it harder for new users to get started without a clear abstraction layer over these components.

### Conclusion:
By incorporating optimizations like memory pooling, JIT compilation, operator fusion, advanced slicing, and thread safety improvements, the system could become much more efficient and versatile. Additionally, features like multi-device support, sparse tensor operations, and seamless cross-device computation would significantly improve the flexibility and scalability of the framework, making it more aligned with state-of-the-art tensor computation libraries.

# Tensor & TensorBase Design

Below is the **hybrid blueprint** combining templates for performance with an internal `DType` variable for flexibility.

### Overview:
- **Template classes** (`Tensor<T>`) for type-specific behavior and performance.
- **Base class** (`TensorBase`) to provide a common interface for all tensor types.
- **`DType` enum** to handle dynamic dispatch based on runtime data type.

---

## 1. **`DType` Enum** (Defines supported types)

```cpp
enum class DType {
    INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64
};
```

---

## 2. **`TensorBase` Class** (Common interface for all tensor types)

```cpp
class TensorBase {
public:
    virtual ~TensorBase() {}

    // Common interface methods
    virtual DType dtype() const = 0; // Get the data type
    virtual DeviceType device_type() const = 0; // Get device type (CPU, GPU, etc.)
    virtual Layout layout() const = 0; // Get memory layout (shape, strides)

    // Optional: common tensor operations (virtual)
    virtual void add(const TensorBase& other, TensorBase& result) const = 0;
    virtual void sub(const TensorBase& other, TensorBase& result) const = 0;
    virtual void mul(const TensorBase& other, TensorBase& result) const = 0;
    virtual void div(const TensorBase& other, TensorBase& result) const = 0;
};
```

---

## 3. **`Tensor<T>` Class** (Concrete class for specific types)

```cpp
template<typename T>
class Tensor : public TensorBase {
public:
    Layout layout;
    DType dtype_value = get_dtype<T>(); // Using helper to assign dtype
    Device* device;
    DeviceType device_type;
    Memory<T>* storage;

    // Constructor, assign dtype from template type
    Tensor(Device* device, DeviceType dtype, Layout layout)
        : device(device), device_type(dtype), layout(layout) {
            // allocate memory or set up layout here
    }

    // Type-specific method implementations
    DType dtype() const override { return dtype_value; }
    DeviceType device_type() const override { return device_type; }
    Layout layout() const override { return layout; }

    // Specific ops on template data type
    void add(const TensorBase& other, TensorBase& result) const override {
        // Implement add for specific dtype
    }

    void sub(const TensorBase& other, TensorBase& result) const override {
        // Implement sub for specific dtype
    }

    void mul(const TensorBase& other, TensorBase& result) const override {
        // Implement mul for specific dtype
    }

    void div(const TensorBase& other, TensorBase& result) const override {
        // Implement div for specific dtype
    }

private:
    // Helper function to get the correct DType from template type
    template<typename T>
    DType get_dtype() const {
        if constexpr (std::is_same<T, int8_t>::value) return DType::INT8;
        else if constexpr (std::is_same<T, int16_t>::value) return DType::INT16;
        else if constexpr (std::is_same<T, int32_t>::value) return DType::INT32;
        else if constexpr (std::is_same<T, int64_t>::value) return DType::INT64;
        else if constexpr (std::is_same<T, float>::value) return DType::FLOAT32;
        else if constexpr (std::is_same<T, double>::value) return DType::FLOAT64;
        else return DType::FLOAT32; // Default (you can expand for others)
    }
};
```

---

## 4. **`OpRegister` Class** (For runtime operation dispatch)

```cpp
class OpRegister {
public:
    using Func = std::function<void(const TensorBase&, const TensorBase&, TensorBase&)>;

    void register_op(const std::string& op_name, DType dtype, DeviceType device_type, Func func) {
        op_map[op_name][dtype][device_type] = func;
    }

    Func get_op(const std::string& op_name, DType dtype, DeviceType device_type) {
        return op_map[op_name][dtype][device_type];
    }

private:
    std::map<std::string, std::map<DType, std::map<DeviceType, Func>>> op_map;
};
```

### Example Operation Registration:

```cpp
OpRegister op_register;

// Register a function for the add operation on CPU and FLOAT32
op_register.register_op("add", DType::FLOAT32, DeviceType::CPU, 
    [](const TensorBase& a, const TensorBase& b, TensorBase& result) {
        // Perform addition (example)
        const Tensor<float>& ta = static_cast<const Tensor<float>&>(a);
        const Tensor<float>& tb = static_cast<const Tensor<float>&>(b);
        Tensor<float>& tr = static_cast<Tensor<float>&>(result);
        
        // Element-wise add
        tr.storage->data = ta.storage->data + tb.storage->data;
    });
```

---

## 5. **`GraphNode` Class** (For storing the computation graph)

```cpp
class GraphNode {
public:
    std::string op_name;
    std::vector<TensorBase*> inputs;
    std::vector<TensorBase*> outputs;

    std::function<void()> compute; // Operation lambda

    GraphNode(const std::string& name, const std::vector<TensorBase*>& in, const std::vector<TensorBase*>& out)
        : op_name(name), inputs(in), outputs(out) {
            for (auto* t : outputs) {
                t->producer = this;
            }
        }

    void execute() {
        if (compute) compute();
    }
};
```

---

## 6. **Graph Class** (Manage computation nodes and execution)

```cpp
class Graph {
public:
    std::vector<GraphNode*> nodes;

    void add_node(GraphNode* node) {
        nodes.push_back(node);
    }

    void execute() {
        for (auto* node : nodes) {
            node->execute();  // Run node computations in topological order
        }
    }
};
```

---

## 🏗️ **Example Workflow**

```cpp
// Create Tensors
Tensor<float>* a = new Tensor<float>(device, DeviceType::CPU, Layout{2, 3});
Tensor<float>* b = new Tensor<float>(device, DeviceType::CPU, Layout{2, 3});
Tensor<float>* result = new Tensor<float>(device, DeviceType::CPU, Layout{2, 3});

// Register operation (Add)
OpRegister op_register;
op_register.register_op("add", DType::FLOAT32, DeviceType::CPU, 
    [](const TensorBase& a, const TensorBase& b, TensorBase& result) {
        const Tensor<float>& ta = static_cast<const Tensor<float>&>(a);
        const Tensor<float>& tb = static_cast<const Tensor<float>&>(b);
        Tensor<float>& tr = static_cast<Tensor<float>&>(result);
        tr.storage->data = ta.storage->data + tb.storage->data;
    });

// Create GraphNode for add
auto node = new GraphNode("add", {a, b}, {result});
node->compute = [=]() {
    auto add_func = op_register.get_op("add", DType::FLOAT32, DeviceType::CPU);
    add_func(*a, *b, *result);  // Execute add operation
};

// Create graph and add node
Graph g;
g.add_node(node);

// Execute computation
g.execute();
```

---

## 🛠️ **Key Takeaways**
- **Performance:** Type-specific operations (`Tensor<T>`) with compile-time safety.
- **Flexibility:** `TensorBase*` allows handling multiple tensor types dynamically in `GraphNode` and `OpRegister`.
- **Runtime dispatching:** `OpRegister` stores operations for each combination of `DType` and `DeviceType`, enabling seamless execution across different backends.

