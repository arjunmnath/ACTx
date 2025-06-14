
#include "device_type.h"
#include "ilcs/py_devices.h"
#include "ilcs/py_tensor.h"
#include "ilcs/py_types.h"
#include "tensor.h"
#include <Python.h>
#include <unordered_map>
#include <vector>

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

PyMODINIT_FUNC PyInit_extension(void) {
  PyObject *module = PyModule_Create(&extension);
  if (module == NULL) {
    return NULL;
  }

  std::unordered_map<std::string, PyObject *> submodules = {
      {"devices", createDevicesModule(module)},
      {"dtype", createDtypeModule(module)},
      {"tensor", createTensorModule(module)}};
  for (const auto &submodule : submodules) {
    if (PyModule_AddObject(module, submodule.first.c_str(), submodule.second) <
        0) {
      Py_DECREF(submodule.second);
      Py_DECREF(module);
    }
  }
  return module;
}
