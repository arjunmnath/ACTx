#pragma once

#include <variant>
using type_variant =
    std::variant<int8_t, int16_t, int32_t, int64_t, _Float16, float>;
enum class DType { int8, int16, int32, int64, float16, float32 };
