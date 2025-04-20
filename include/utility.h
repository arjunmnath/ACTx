
#pragma once
#include "tensor.h"
#include "types.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec);
template <typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs);
float __rand(int seed = -1);
float __randn(float mean = 0, float stddev = 1, int seed = -1);
int __randint(int min, int max, int seed = -1);
int __poisson(float mean, int seed = -1);
int __bernoulli(float p, int seed = -1);
std::vector<int> compute_broadcast_shape(const Tensor &a, const Tensor &b);
int getDTypeSize(DType type);
std::string getDeviceName(DeviceType device);
std::string getTypeName(DType dtype);
