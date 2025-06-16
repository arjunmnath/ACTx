#pragma once
#include <Python.h>
PyObject *createTensorModule(PyObject *parent);
extern PyTypeObject PyTensorType;
