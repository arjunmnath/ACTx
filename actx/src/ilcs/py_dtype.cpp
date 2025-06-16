#include "ilcs/py_types.h"
#include "types.h"

PyObject *createDtypeModule(PyObject *parent) {
  PyObject *dtype = PyModule_New("extension.dtype");
  if (dtype == NULL) {
    Py_DECREF(parent);
    return NULL;
  }
  PyModule_AddIntConstant(dtype, "i8", static_cast<int>(DType::int8));
  PyModule_AddIntConstant(dtype, "i16", static_cast<int>(DType::int16));
  PyModule_AddIntConstant(dtype, "i32", static_cast<int>(DType::int32));
  PyModule_AddIntConstant(dtype, "i64", static_cast<int>(DType::int64));
  PyModule_AddIntConstant(dtype, "f16", static_cast<int>(DType::float16));
  PyModule_AddIntConstant(dtype, "f32", static_cast<int>(DType::float32));

  return dtype;
}
