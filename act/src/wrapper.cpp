
#include "tensor.h"
#include <vector>
extern "C" {
#include <Python.h>
}

static PyObject *add(PyObject *self, PyObject *args) {
  int a, b;
  if (!PyArg_ParseTuple(args, "ii", &a, &b))
    return NULL;

  std::vector<int> shape = {2, 3};
  Tensor _a = Tensor::zeros(shape);
  std::vector<float> ones = {0, 0, 0, 0, 0, 0};
  Tensor expected(ones, shape, DType::int32);
  _a.print();
  expected.print();
  Tensor _b = _a.logical_e(&expected);
  _b.print();

  return PyLong_FromLong(a + b);
}

static PyMethodDef MyMethods[] = {{"add", add, METH_VARARGS, "Add two numbers"},
                                  {NULL, NULL, 0, NULL}};

static struct PyModuleDef extension = {PyModuleDef_HEAD_INIT, "extension",
                                       "Example module", -1, MyMethods};

PyMODINIT_FUNC PyInit_extension(void) { return PyModule_Create(&extension); }
