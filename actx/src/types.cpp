#include "types.h"
#include <string>
#include <vector>

std::string getTypeName(DType dtype) {
  switch (dtype) {
  case DType::int8:
    return "int8";
  case DType::int16:
    return "int16";
  case DType::int32:
    return "int32";
  case DType::int64:
    return "int64";
  case DType::float16:
    return "float16";
  case DType::float32:
    return "float32";
  default:
    return "unknown type";
  }
}
int getDTypeSize(DType dtype) {
  switch (dtype) {
  case DType::int8:
    return 1;
    break;
  case DType::float16:
  case DType::int16:
    return 2;
  case DType::float32:
  case DType::int32:
    return 4;
  case DType::int64:
    return 8;
  default:
    throw std::invalid_argument("not implemented");
    break;
  }
}

void *get_raw_pointer(const std::vector<std::variant<int, float, double>> &vec,
                      DType dtype, size_t &out_size_bytes) {
  size_t count = vec.size();
  out_size_bytes = 0;

  if (vec.empty())
    return nullptr;

  if (dtype == DType::int32) {
    int *buffer = new int[count];
    for (size_t i = 0; i < count; ++i)
      buffer[i] = std::get<int>(vec[i]);
    out_size_bytes = count * sizeof(int);
    return buffer;

  } else if (dtype == DType::float32) {
    float *buffer = new float[count];
    for (size_t i = 0; i < count; ++i)
      buffer[i] = std::get<float>(vec[i]);
    out_size_bytes = count * sizeof(float);
    return buffer;
  }
  return nullptr;
}
