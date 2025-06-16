
#pragma once
#include "tensor.h"
#include "types.h"
#include <any>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define RESET "\033[0m"

// Regular colors
#define BLACK "\033[0;30m"
#define RED "\033[0;31m"
#define GREEN "\033[0;32m"
#define YELLOW "\033[0;33m"
#define BLUE "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN "\033[0;36m"
#define WHITE "\033[0;37m"

// Bold colors
#define BOLD_BLACK "\033[1;30m"
#define BOLD_RED "\033[1;31m"
#define BOLD_GREEN "\033[1;32m"
#define BOLD_YELLOW "\033[1;33m"
#define BOLD_BLUE "\033[1;34m"
#define BOLD_MAGENTA "\033[1;35m"
#define BOLD_CYAN "\033[1;36m"
#define BOLD_WHITE "\033[1;37m"

#define COLOR(str, color) (std::string(color) + std::string(str) + RESET)

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec);
template <typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs);
float __rand(int seed = -1);
float __randn(float mean = 0, float stddev = 1, int seed = -1);
int __randint(int min, int max, int seed = -1);
int __poisson(float mean, int seed = -1);
int __bernoulli(float p, int seed = -1);
std::vector<int> compute_broadcast_shape(const Tensor *a, const Tensor *b);
int getDTypeSize(DType type);
std::string getDeviceName(DeviceType device);
std::string getTypeName(DType dtype);
