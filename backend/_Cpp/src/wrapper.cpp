#include "mps.h"
#include <Python.h>
#include <vector>

typedef struct {
  PyObject_HEAD MPS *cpp_obj;
} PyMPS;

static int PyMPS_init(PyMPS *self, PyObject *args, PyObject *kwds) {
  self->cpp_obj = new MPS();
  return 0;
}

static void PyMPS_dealloc(PyMPS *self) {
  delete self->cpp_obj;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyMPS_run(PyMPS *self, PyObject *args) {
  bool isTransponse;
  if (!PyArg_ParseTuple(args, "p", &isTransponse)) {
    return NULL;
  }
  std::vector<int> view = {5, 1};
  std::vector<int> transpose = {1, 5};

  std::vector<id<MTLBuffer>> buffers = self->cpp_obj->__dummy_data();
  self->cpp_obj->execute_kernel_binary("elementwise_multiply_matrix",
                                       buffers[0], buffers[1], buffers[2],
                                       buffers[3]);
  if (isTransponse) {
    self->cpp_obj->print_buffer_contents(buffers, transpose);
  } else {
    self->cpp_obj->print_buffer_contents(buffers, view);
  }
  Py_RETURN_NONE;
}

static PyMethodDef PyMPS_methods[] = {
    {"run", (PyCFunction)PyMPS_run, METH_VARARGS, "Dummy Run Method"},
    {NULL, NULL, 0, NULL} // Sentinel (indicate end of array)
};

static PyTypeObject PyMPSType = {
    PyVarObject_HEAD_INIT(NULL, 0) "MPS.mps", // tp_name
    sizeof(PyMPS),                            // tp_basicsize
    0,                                        // tp_itemsize
    (destructor)PyMPS_dealloc,                // tp_dealloc
    0,                                        // tp_print
    0,                                        // tp_getattr
    0,                                        // tp_setattr
    0,                                        // tp_as_async
    0,                                        // tp_repr
    0,                                        // tp_as_number
    0,                                        // tp_as_sequence
    0,                                        // tp_as_mapping
    0,                                        // tp_hash
    0,                                        // tp_call
    0,                                        // tp_str
    0,                                        // tp_getattro
    0,                                        // tp_setattro
    0,                                        // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                       // tp_flags
    // TODO: write docs
    "nil",                // tp_doc
    0,                    // tp_traverse
    0,                    // tp_clear
    0,                    // tp_richcompare
    0,                    // tp_weaklistoffset
    0,                    // tp_iter
    0,                    // tp_iternext
    PyMPS_methods,        // tp_methods
    0,                    // tp_members
    0,                    // tp_getset
    0,                    // tp_base
    0,                    // tp_dict
    0,                    // tp_descr_get
    0,                    // tp_descr_set
    0,                    // tp_dictoffset
    (initproc)PyMPS_init, // tp_init
    0,                    // tp_alloc
    PyType_GenericNew,    // tp_new
};

static PyModuleDef MPSModule = {PyModuleDef_HEAD_INIT,
                                "MPS",                       // Module name
                                "A mps opeartions testings", // Module docstring
                                -1,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL};

// Module initialization
PyMODINIT_FUNC PyInit_mps(void) {
  PyObject *m;

  if (PyType_Ready(&PyMPSType) < 0)
    return NULL;

  m = PyModule_Create(&MPSModule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&PyMPSType);
  PyModule_AddObject(m, "mps", (PyObject *)&PyMPSType);
  return m;
}
