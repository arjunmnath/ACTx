#include "utility.h"
#include "tensor.h"
#include "types.h"
#include <cstdint>
#include <iostream>
#include <random>

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

int getDTypeSize(DType type) {
  switch (type) {
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
  case DType::float64:
  case DType::int64:
    return 8;
    break;
  default:
    throw std::invalid_argument("not implemented");
    break;
  }
}

std::vector<int> compute_broadcast_shape(const Tensor &a, const Tensor &b) {
  int max_rank = std::max(b.dims.size(), a.dims.size());

  std::vector<int> rev_shape1 = a.dims;
  std::vector<int> rev_shape2 = b.dims;

  std::reverse(rev_shape1.begin(), rev_shape1.end());
  std::reverse(rev_shape2.begin(), rev_shape2.end());

  rev_shape1.resize(max_rank, 1);
  rev_shape2.resize(max_rank, 1);

  std::vector<int> result(max_rank);

  int dim1, dim2;
  for (int i = 0; i < max_rank; i++) {
    dim1 = rev_shape1[i];
    dim2 = rev_shape2[i];
    if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
      result[i] = std::max(dim1, dim2);
    } else {
      throw std::invalid_argument("Shapes not broadcastable");
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}
