
#include "device_type.h"
#include "ilcs/py_devices.h"
#include "ilcs/py_tensor.h"
#include "ilcs/py_types.h"
#include "tensor.h"
#include <Python.h>
#include <unordered_map>
#include <vector>

static PyMethodDef MyMethods[] = {{NULL, NULL, 0, NULL}};
static struct PyModuleDef extension = {PyModuleDef_HEAD_INIT, "extension",
                                       "Wrapper module", -1, MyMethods};

PyMODINIT_FUNC PyInit_extension(void) {
  PyObject *module = PyModule_Create(&extension);
  if (module == NULL) {
    return NULL;
  }

  std::unordered_map<std::string, PyObject *> submodules = {
      {"devices", createDevicesModule(module)},
      {"dtype", createDtypeModule(module)},
      {"tensor", createTensorModule(module)},
  };
  for (const auto &submodule : submodules) {
    if (PyModule_AddObject(module, submodule.first.c_str(), submodule.second) <
        0) {
      Py_DECREF(submodule.second);
      Py_DECREF(module);
    }
  }
  return module;
}
