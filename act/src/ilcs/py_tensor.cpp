#include "ilcs/py_tensor.h"
#include "object.h"
#include "tensor.h"
#include "types.h"
#include <memory>

extern PyTypeObject PyTensorType;
typedef struct {
  Tensor *_native_obj;
} TensorStruct;

typedef struct {
  PyObject_HEAD TensorStruct *inner;
} PyTensorObject;

void TensorInitdims(TensorStruct *self, std::vector<int> dims, int DtypeInt,
                    bool requires_grad) {
  self->_native_obj =
      new Tensor(dims, static_cast<DType>(DtypeInt), requires_grad);
}

static void PyTensor_dealloc(PyTensorObject *self) {
  delete self->inner->_native_obj;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyTensor_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwargs) {
  PyTensorObject *self = (PyTensorObject *)type->tp_alloc(type, 0);
  if (self == NULL)
    return NULL;

  self->inner = new TensorStruct();
  if (self->inner == NULL) {
    Py_DECREF(self);
    return NULL;
  }
  return (PyObject *)self;
}

static int PyTensor_init(PyTensorObject *self, PyObject *args,
                         PyObject *kwargs) {
  PyObject *first = nullptr;
  int DTypeInt = static_cast<int>(DType::float32);
  bool requires_grad = false;
  static const char *keywords[] = {"dims", "dtype", "requires_grad", NULL};

  if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|ip", (char **)keywords,
                                  &first, &DTypeInt, &requires_grad)) {

    // shape bases init
    if (PyTuple_Check(first)) {
      Py_ssize_t ndim = PyTuple_Size(first);
      if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "dims must not be empty");
        return -1;
      }
      std::vector<int> dims;
      dims.reserve(ndim);
      for (Py_ssize_t i = 0; i < ndim; ++i) {
        PyObject *item = PyTuple_GetItem(first, i);
        if (!PyLong_Check(item)) {
          PyErr_SetString(PyExc_TypeError, "dims must be integers");
          return -1;
        }
        long dim = PyLong_AsLong(item);
        if (dim <= 0) {
          PyErr_SetString(PyExc_ValueError, "dims must be positive integers");
          return -1;
        }
        dims.push_back(static_cast<int>(dim));
      }
      TensorInitdims(self->inner, dims, DTypeInt, requires_grad);
    } else if (PyList_Check(first)) {
      throw std::runtime_error("not implemented");
      // } else if (PyArray_Check(first)) {
      //   throw std::runtime_error("not implemented");
    }
    return 0;
  }
  PyErr_SetString(PyExc_TypeError, "Invalid arguments.");
  return -1;
}

static PyObject *PyTensor_print(PyTensorObject *self,
                                PyObject *Py_UNUSED(ignored)) {
  self->inner->_native_obj->print();
  return Py_None;
}

static PyObject *PyTensor_print_buffer(PyTensorObject *self,
                                       PyObject *Py_UNUSED(ignored)) {
  self->inner->_native_obj->print_buffer();
  return Py_None;
}

// ────────────────────────────────────────────
// Field Getters
// ────────────────────────────────────────────
static PyObject *PyTensor_get_requires_grad(PyTensorObject *self,
                                            void *closure) {
  return PyBool_FromLong(self->inner->_native_obj->requires_grad ? 1 : 0);
}

// ────────────────────────────────────────────
// Field Setters
// ────────────────────────────────────────────
static int PyTensor_set_requires_grad(PyTensorObject *self, PyObject *value,
                                      void *closure) {
  if (!PyBool_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "Expected Boolean Value.");
    return -1;
  }
  self->inner->_native_obj->requires_grad = PyObject_IsTrue(value) == 1;
  return 0;
}

// ────────────────────────────────────────────
// Arithemetic operators
// ────────────────────────────────────────────

static PyObject *PyTensor_add(PyObject *a, PyObject *b) {
  if (!PyObject_TypeCheck(a, &PyTensorType) ||
      !PyObject_TypeCheck(b, &PyTensorType)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  PyTensorObject *res_obj = PyObject_New(PyTensorObject, &PyTensorType);
  if (!res_obj)
    return NULL;
  res_obj->inner->_native_obj =
      ((PyTensorObject *)a)
          ->inner->_native_obj->add(((PyTensorObject *)b)->inner->_native_obj,
                                    false);

  return (PyObject *)res_obj;
}

static PyObject *PyTensor_add_inplace(PyObject *a, PyObject *b) {
  if (!PyObject_TypeCheck(a, &PyTensorType) ||
      !PyObject_TypeCheck(b, &PyTensorType)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  ((PyTensorObject *)a)
      ->inner->_native_obj->add(((PyTensorObject *)b)->inner->_native_obj,
                                true);

  return (PyObject *)a;
}

static PyMethodDef PyTensor_methods[] = {
    {"print", (PyCFunction)PyTensor_print, METH_NOARGS, "Print the tensor"},
    {"print_buffer", (PyCFunction)PyTensor_print_buffer, METH_NOARGS,
     "Print the Tensor Buffer"},
    {NULL}};

static PyGetSetDef PyTensor_getsets[] = {
    {"requires_grad", (getter)PyTensor_get_requires_grad,
     (setter)PyTensor_set_requires_grad,
     "Boolean flag indicating whether this tensor should track operations for "
     "gradient computation.\n"
     "When set to True, the tensor records operations to enable automatic "
     "differentiation during backpropagation.\n"
     "Defaults to False.",
     NULL},

    {NULL}};

static PyNumberMethods PyTensor_as_number = {
    .nb_add = PyTensor_add,
    .nb_inplace_add = (binaryfunc)PyTensor_add_inplace,
};
PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "extension.tensor.Tensor",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_dealloc = (destructor)PyTensor_dealloc,
    .tp_as_number = &PyTensor_as_number,
    .tp_methods = PyTensor_methods,
    .tp_getset = PyTensor_getsets,
    .tp_init = (initproc)PyTensor_init,
    .tp_new = PyTensor_new,
};

PyObject *createTensorModule(PyObject *parent) {
  PyObject *tensor = PyModule_New("extension.tensor");
  if (tensor == NULL) {
    Py_DECREF(parent);
    return NULL;
  }
  if (PyType_Ready(&PyTensorType) < 0) {
    Py_DECREF(tensor);
    Py_DECREF(parent);
    return NULL;
  }

  Py_INCREF((PyObject *)&PyTensorType);
  if (PyModule_AddObject(tensor, "Tensor", (PyObject *)&PyTensorType) < 0) {
    Py_DECREF((PyObject *)&PyTensorType);
    Py_DECREF(tensor);
    Py_DECREF(parent);
    return NULL;
  }

  return tensor;
}
