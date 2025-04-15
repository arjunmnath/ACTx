#include "utility.h"
#include "types.h"
#include <cstdint>
#include <iostream>
#include <random>

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

float  __randn(float mean, float stddev, int seed) {
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

int __poisson(void * p, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::poisson_distribution<int> poisson(*static_cast<float*>(p));
  return poisson(gen);
}

int __bernoulli(void * p, int seed) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
    std::bernoulli_distribution dist(*static_cast<float*>(p));
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
