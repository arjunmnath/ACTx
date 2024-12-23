#include "mps_ops.h"
#include <Python.h>
#include <vector>
/*
  Stack();
  void push(int value);
  int pop();
  int top();
  bool is_empty();
  string to_string();
 std::vector<int> data;
*/

// Python wrapper object for Stack
typedef struct {
  PyObject_HEAD Stack *cpp_obj; // Pointer to the C++ object
} PyStack;

// Constructor for the Python wrapper
static int PyStack_init(PyStack *self, PyObject *args, PyObject *kwds) {
  self->cpp_obj = new Stack(); // Create the C++ object
  return 0;
}

// Destructor for the Python wrapper
static void PyStack_dealloc(PyStack *self) {
  delete self->cpp_obj; // Delete the C++ object
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Method: push
static PyObject *PyStack_push(PyStack *self, PyObject *args) {
  int value;
  if (!PyArg_ParseTuple(args, "i", &value)) {
    return NULL;
  }
  self->cpp_obj->push(value);
  y_RETURN_NONE;
}

// Method: pop
static PyObject *PyStack_pop(PyStack *self, PyObject *Py_UNUSED(ignored)) {
  try {
    int value = self->cpp_obj->pop();
    return PyLong_FromLong(value);
  } catch (const std::runtime_error &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

// Method: top
static PyObject *PyStack_top(PyStack *self, PyObject *Py_UNUSED(ignored)) {
  try {
    int value = self->cpp_obj->top();
    return PyLong_FromLong(value);
  } catch (const std::runtime_error &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }
}

// Method: is_empty
static PyObject *PyStack_is_empty(PyStack *self, PyObject *Py_UNUSED(ignored)) {
  return PyBool_FromLong(self->cpp_obj->is_empty());
}

// Method: to_string
static PyObject *PyStack_to_string(PyStack *self,
                                   PyObject *Py_UNUSED(ignored)) {
  std::string result = self->cpp_obj->to_string();
  return PyUnicode_FromString(result.c_str());
}

// Method definitions
static PyMethodDef PyStack_methods[] = {
    {"push", (PyCFunction)PyStack_push, METH_VARARGS,
     "Push a value onto the stack"},
    {"pop", (PyCFunction)PyStack_pop, METH_NOARGS,
     "Pop a value from the stack"},
    {"top", (PyCFunction)PyStack_top, METH_NOARGS,
     "Get the top value of the stack"},
    {"is_empty", (PyCFunction)PyStack_is_empty, METH_NOARGS,
     "Check if the stack is empty"},
    {"to_string", (PyCFunction)PyStack_to_string, METH_NOARGS,
     "Get a string representation of the stack"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Python type definition
static PyTypeObject PyStackType = {
    PyVarObject_HEAD_INIT(NULL, 0) "StackModule.Stack", // tp_name
    sizeof(PyStack),                                    // tp_basicsize
    0,                                                  // tp_itemsize
    (destructor)PyStack_dealloc,                        // tp_dealloc
    0,                                                  // tp_print
    0,                                                  // tp_getattr
    0,                                                  // tp_setattr
    0,                                                  // tp_as_async
    0,                                                  // tp_repr
    0,                                                  // tp_as_number
    0,                                                  // tp_as_sequence
    0,                                                  // tp_as_mapping
    0,                                                  // tp_hash
    0,                                                  // tp_call
    0,                                                  // tp_str
    0,                                                  // tp_getattro
    0,                                                  // tp_setattro
    0,                                                  // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                                 // tp_flags
    "Stack objects",                                    // tp_doc
    0,                                                  // tp_traverse
    0,                                                  // tp_clear
    0,                                                  // tp_richcompare
    0,                                                  // tp_weaklistoffset
    0,                                                  // tp_iter
    0,                                                  // tp_iternext
    PyStack_methods,                                    // tp_methods
    0,                                                  // tp_members
    0,                                                  // tp_getset
    0,                                                  // tp_base
    0,                                                  // tp_dict
    0,                                                  // tp_descr_get
    0,                                                  // tp_descr_set
    0,                                                  // tp_dictoffset
    (initproc)PyStack_init,                             // tp_init
    0,                                                  // tp_alloc
    PyType_GenericNew,                                  // tp_new
};

// Module definition
static PyModuleDef StackModule = {
    PyModuleDef_HEAD_INIT,
    "StackModule",                                  // Module name
    "A Python module that wraps a C++ Stack class", // Module docstring
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL};

// Module initialization
PyMODINIT_FUNC PyInit_StackModule(void) {
  PyObject *m;

  if (PyType_Ready(&PyStackType) < 0)
    return NULL;

  m = PyModule_Create(&StackModule);
  if (m == NULL)
    return NULL;

  Py_INCREF(&PyStackType);
  PyModule_AddObject(m, "Stack", (PyObject *)&PyStackType);
  return m;
}
