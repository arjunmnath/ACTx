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
template <typename T> T __rand(int seed = -1) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> uniform_dist(0, 1);
  return static_cast<T>(uniform_dist(gen));
}

template <typename T>
T __randn(float mean = 0, float stddev = 1, int seed = -1) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::normal_distribution<T> normal(mean, stddev);
  return static_cast<T>(normal(gen));
}
int __randint(int min, int max, int seed = -1) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> int_dist(min, max - 1);
  return int_dist(gen);
}

template <typename T> int __poisson(T p, int seed = -1) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::poisson_distribution<int> poisson(p);
  return poisson(gen);
}

template <typename T> int __bernoulli(T p, int seed = -1) {
  if (-1 == seed) {
    std::random_device rd;
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::bernoulli_distribution dist(p);
  return dist(gen);
}
