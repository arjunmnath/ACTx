

- [ ] use MTLHeap instead of MTLBuffer on memory pooling
- [ ] avoid using shared memory modes and use private for computationally heavy tensors
- [ ] use a template type for metal kernels to support datatype like
   | Type   | Description              |
  |--------|--------------------------|
  | `char`   | 8-bit signed integer     |
  | `uchar`  | 8-bit unsigned integer   |
  | `short`  | 16-bit signed integer    |
  | `ushort` | 16-bit unsigned integer  |
  | `int`    | 32-bit signed integer    |
  | `uint`   | 32-bit unsigned integer  |
  | `half`   | 16-bit floating point    |
  | `float`  | 32-bit floating point    |
  | `not supported`| 64-bit floating point |
  | `bool`   | Boolean                  |

