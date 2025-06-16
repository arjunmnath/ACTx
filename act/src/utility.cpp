#include "utility.h"
#include "tensor.h"
#include "types.h"
#include <any>
#include <cstdint>
#include <iostream>
#include <random>
#include <variant>

// TODO: FIX THE HARD CODED TYPE MANAGEMENT IN bernoulli poisson etc
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << static_cast<T>(vec[i]);
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
template <typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
}
float __rand(int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> uniform_dist(0, 1);
  return static_cast<float>(uniform_dist(gen));
}

float __randn(float mean, float stddev, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(mean, stddev);
  return static_cast<float>(normal(gen));
}
int __randint(int min, int max, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> int_dist(min, max - 1);
  return int_dist(gen);
}

int __poisson(float mean, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::poisson_distribution<int> poisson(mean);
  return poisson(gen);
}

int __bernoulli(float p, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::bernoulli_distribution dist(p);
  return dist(gen);
}

int getDTypeSize(DType dtype) {
  switch (dtype) {
  case DType::int8:
    return 1;
    break;
  case DType::float16:
  case DType::int16:
    return 2;
    break;

  case DType::float32:
  case DType::int32:
    return 4;
    break;
  case DType::int64:
    return 8;
    break;
  default:
    throw std::invalid_argument("not implemented");
    break;
  }
}
std::string getDeviceName(DeviceType device) {
  switch (device) {
  case DeviceType::MPS:
    return "MPS";
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::WEBGPU:
    return "WEBGPU";
  default:
    return "unknown device";
  }
}
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
// TODO: complete remaining data types
std::vector<int> compute_broadcast_shape(const Tensor *a, const Tensor *b) {
  int max_rank = std::max(b->dims.size(), a->dims.size());
  std::vector<int> result(max_rank);
  for (int i = 0; i < max_rank; ++i) {
    int dim1 = (i < a->dims.size()) ? a->dims[(a->dims.size() - 1) - i] : 1;
    int dim2 = (i < b->dims.size()) ? b->dims[(b->dims.size() - 1) - i] : 1;
    if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
      result[(max_rank - 1) - i] = std::max(dim1, dim2);
    else
      throw std::invalid_argument("Shapes not broadcastable");
  }
  return result;
}
