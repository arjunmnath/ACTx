
#pragma once
#include "types.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int getDTypeSize(DType type);

template <typename T> T __rand(int seed = -1);
template <typename T>
T __randn(float mean = 0, float stddev = 1, int seed = -1);
int __randint(int min, int max, int seed = -1);
template <typename T> int __poisson(T p, int seed = -1);
template <typename T> int __bernoulli(T p, int seed = -1);

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec);

template <typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs);
