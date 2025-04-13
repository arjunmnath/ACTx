#pragma once 

#include <mutex>
#include <string>
#include "device_type.h"
template <typename T>
class Memory {
private:
    T memory;
    std::mutex _lock;
    DeviceType _type;

public:
    void aquire_lock();
    void release_lock();
    void guared_lock();
};

