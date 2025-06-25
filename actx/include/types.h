#pragma once

#include <variant>
#include <vector>
using type_t = std::variant<int8_t, int16_t, int32_t, int64_t, _Float16, float>;
using types_pointer_t = std::variant<int8_t *, int16_t *, int32_t *, int64_t *,
                                     _Float16 *, float *>;

enum class DType { int8, int16, int32, int64, float16, float32 };

std::string getTypeName(DType dtype);

int getDTypeSize(DType dtype);
void *get_raw_pointer(const std::vector<std::variant<int, float, double>> &vec,
                      DType dtype, size_t &out_size_bytes);
