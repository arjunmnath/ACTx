#pragma once

#include <string>
#include "memory.h"

class Device {
private:
    std::string _name;

public:
    template <typename T>
    Memory<T> alloc();
    void sync();
    std::string name() {
        return this->_name;
    }
};

