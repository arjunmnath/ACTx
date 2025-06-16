# TODOS

- [ ] use MTLHeap instead of MTLBuffer on memory pooling
- [ ] avoid using shared memory modes and use private for computationally heavy tensors
- [ ] use a template type for metal kernels to support datatype like

  | Type     | Description             | Platfrom support               |
  | -------- | ----------------------- | ------------------------------ |
  | `char`   | 8-bit signed integer    | `mps`, `cuda`, `cpu`           |
  | `uchar`  | 8-bit unsigned integer  | `mps`, `cuda`, `cpu`, `webgpu` |
  | `short`  | 16-bit signed integer   | `mps`, `cuda`, `cpu`, `webgpu` |
  | `ushort` | 16-bit unsigned integer | `mps`, `cuda`, `cpu`, `webgpu` |
  | `int`    | 32-bit signed integer   | `mps`, `cuda`, `cpu`, `webgpu` |
  | `uint`   | 32-bit unsigned integer | `mps`, `cuda`, `cpu`, `webgpu` |
  | `half`   | 16-bit floating point   | `mps`, `cuda`, `webgpu`        |
  | `float`  | 32-bit floating point   | `mps`, `cuda`, `cpu`, `webgpu` |
  | `double` | 64-bit floating point   | `cuda`, `cpu`                  |
  | `bool`   | Boolean                 | `mps`, `cuda`, `cpu`, `webgpu` |

- [ ] change usage of shared_ptr to weak_ptr wherever possible
- [ ] use open mp to implement cpu kernels
- [ ] enable mutex locks for memory
- [ ] copying meta data to a buffer for every kernel operation is expensive, fix that;
- [ ] simplify the compute_broadcast_index kernel helper logic
- [ ] operations are not allowed on views currently, fix that
- [x] use a different thread allocation logic (groups = 1, thread per group = maxThreadsAvailable if size < maxThreadsAvailable
      else groups = size // maxThreadsAvailable, thread per group = maxThreadsAvailable)
