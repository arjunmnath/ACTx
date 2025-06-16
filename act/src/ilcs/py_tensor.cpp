#include "ilcs/py_tensor.h"
#include "floatobject.h"
#include "longobject.h"
#include "numpy/ndarrayobject.h"
#include "object.h"
#include "tensor.h"
#include "types.h"
#include <iostream>
#include <memory>
#include <numpy/arrayobject.h>

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

  import_array();
  if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|ip", (char **)keywords,
                                  &first, &DTypeInt, &requires_grad)) {

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
      return 0;
    } else if (first && PyArray_Check(first)) {
      /*
       * NPY_BOOL          // boolean
       * NPY_BYTE          // signed 8-bit integer (int8)
       * NPY_UBYTE         // unsigned 8-bit integer (uint8)
       * NPY_SHORT         // signed 16-bit integer (int16)
       * NPY_USHORT        // unsigned 16-bit integer (uint16)
       * NPY_INT           // signed 32-bit integer (int32)
       * NPY_UINT          // unsigned 32-bit integer (uint32)
       * NPY_LONG          // signed long (platform dependent, usually int64 on
       * 64-bit) NPY_ULONG         // unsigned long (platform dependent)
       * NPY_LONGLONG      // signed 64-bit integer (int64)
       * NPY_ULONGLONG     // unsigned 64-bit integer (uint64)
       * NPY_FLOAT         // float32 (single precision)
       * NPY_DOUBLE        // float64 (double precision)
       * NPY_LONGDOUBLE    // extended precision float (usually 80 or 128 bit)
       * NPY_CFLOAT        // complex64 (complex with float32 real and imag)
       * NPY_CDOUBLE       // complex128 (complex with float64 real and imag)
       * NPY_CLONGDOUBLE   // complex extended precision
       * NPY_OBJECT        // Python objects
       * NPY_STRING        // raw byte strings
       * NPY_UNICODE       // unicode strings
       * NPY_VOID          // raw data (structured or void type)
       */

      PyArrayObject *array = (PyArrayObject *)first;
      if (PyArray_TYPE(array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "expected np.float32 array");
        return -1;
      }
      int ndim = PyArray_NDIM(array);
      npy_intp *np_shape = PyArray_SHAPE(array);
      std::vector<int> shape;
      std::vector<float> values;
      shape.reserve(ndim);
      for (int i = 0; i < ndim; ++i) {
        shape.push_back((int)np_shape[i]);
      }
      int size = PyArray_SIZE(array);
      float *array_data = (float *)PyArray_DATA(array);
      values.assign(array_data, array_data + size);

      self->inner->_native_obj = new Tensor(values, shape);
      return 0;
    } else if (PyList_Check(first)) {
      return -1;
    } else {
      PyErr_SetString(PyExc_TypeError, "Invalid arguments.");
      return -1;
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "Invalid arguments.");
    return -1;
  }
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
// Helper Methods
// ────────────────────────────────────────────
Tensor *compute_hoist(PyObject *a, PyObject *b) {
  if (PyObject_TypeCheck(a, &PyTensorType)) {
    if (PyFloat_Check(b)) {
      double val = PyFloat_AsDouble(b);
      std::vector<int> shape = {1};
      std::vector<float> values = {(float)val};
      return new Tensor(
          values, shape, DType::float32,
          ((PyTensorObject *)a)->inner->_native_obj->requires_grad);
    }
    // } else if (PyLong_Check(b)) {
    //   long val = PyLong_AsLong(b);
    //   std::vector<int> shape = {1};
    //   std::vector<int> values = {(int)val};
    //   return new Tensor(
    //       values, shape, DType::int32,
    //       ((PyTensorObject *)a)->inner->_native_obj->requires_grad);
    // }
    else if (PyObject_TypeCheck(b, &PyTensorType)) {
      return ((PyTensorObject *)b)->inner->_native_obj;
    } else {
      PyErr_SetString(
          PyExc_TypeError,
          "Invalid argument type, Expected int, float or Tensor object");
      return NULL;
    }
  } else if (PyObject_TypeCheck(b, &PyTensorType)) {
    if (PyFloat_Check(a)) {
      double val = PyFloat_AsDouble(a);
      std::vector<int> shape = {1};
      std::vector<float> values = {(float)val};
      return new Tensor(
          values, shape, DType::float32,
          ((PyTensorObject *)b)->inner->_native_obj->requires_grad);
    }
    // else if (PyLong_Check(a)) {
    //   long val = PyLong_AsLong(a);
    //   std::vector<int> shape = {1};
    //   std::vector<int> values = {(int)val};
    //   return new Tensor(
    //       values, shape, DType::int32,
    //       ((PyTensorObject *)b)->inner->_native_obj->requires_grad);
    // }
    else {
      PyErr_SetString(
          PyExc_TypeError,
          "Invalid argument type, Expected int, float or Tensor object");
      return NULL;
    }
  } else {
    PyErr_SetString(
        PyExc_TypeError,
        "Invalid argument type, Expected int, float or Tensor object");
    return NULL;
  }
}
// ────────────────────────────────────────────
// Arithemetic operators
// ────────────────────────────────────────────
static PyObject *PyTensor_add(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (other == NULL) {
    return NULL;
  }

  PyTensorObject *res_obj = PyObject_New(PyTensorObject, &PyTensorType);
  if (!res_obj)
    return NULL;
  std::vector<int> shape = {4, 4};
  res_obj->inner = new TensorStruct;
  res_obj->inner->_native_obj =
      ((PyTensorObject *)a)->inner->_native_obj->add(other, false);
  return (PyObject *)res_obj;
}

static PyObject *PyTensor_add_inplace(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (!other) {
    return NULL;
  }
  Tensor *out = ((PyTensorObject *)a)->inner->_native_obj->add(other, true);
  if (!out) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "in-place operation is not allowed on a tensor that requires "
        "gradient. "
        "Please detach the tensor or clone it before proceeding.");
    return NULL;
  }
  Py_INCREF(a);
  return (PyObject *)a;
}

static PyObject *PyTensor_sub(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (other == NULL) {
    return NULL;
  }

  PyTensorObject *res_obj = PyObject_New(PyTensorObject, &PyTensorType);
  if (!res_obj)
    return NULL;
  std::vector<int> shape = {4, 4};
  res_obj->inner = new TensorStruct;
  res_obj->inner->_native_obj =
      ((PyTensorObject *)a)->inner->_native_obj->sub(other, false);
  return (PyObject *)res_obj;
}

static PyObject *PyTensor_sub_inplace(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (!other) {
    return NULL;
  }
  Tensor *out = ((PyTensorObject *)a)->inner->_native_obj->sub(other, true);
  if (!out) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "in-place operation is not allowed on a tensor that requires "
        "gradient. "
        "Please detach the tensor or clone it before proceeding.");
    return NULL;
  }
  Py_INCREF(a);
  return (PyObject *)a;
}

static PyObject *PyTensor_div(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (other == NULL) {
    return NULL;
  }

  PyTensorObject *res_obj = PyObject_New(PyTensorObject, &PyTensorType);
  if (!res_obj)
    return NULL;
  std::vector<int> shape = {4, 4};
  res_obj->inner = new TensorStruct;
  res_obj->inner->_native_obj =
      ((PyTensorObject *)a)->inner->_native_obj->div(other, false);
  return (PyObject *)res_obj;
}

static PyObject *PyTensor_div_inplace(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (!other) {
    return NULL;
  }
  Tensor *out = ((PyTensorObject *)a)->inner->_native_obj->div(other, true);
  if (!out) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "in-place operation is not allowed on a tensor that requires "
        "gradient. "
        "Please detach the tensor or clone it before proceeding.");
    return NULL;
  }
  Py_INCREF(a);
  return (PyObject *)a;
}

static PyObject *PyTensor_mul(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (other == NULL) {
    return NULL;
  }

  PyTensorObject *res_obj = PyObject_New(PyTensorObject, &PyTensorType);
  if (!res_obj)
    return NULL;
  std::vector<int> shape = {4, 4};
  res_obj->inner = new TensorStruct;
  res_obj->inner->_native_obj =
      ((PyTensorObject *)a)->inner->_native_obj->mul(other, false);
  return (PyObject *)res_obj;
}

static PyObject *PyTensor_mul_inplace(PyObject *a, PyObject *b) {
  Tensor *other = compute_hoist(a, b);
  if (!other) {
    return NULL;
  }
  Tensor *out = ((PyTensorObject *)a)->inner->_native_obj->mul(other, true);
  if (!out) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "in-place operation is not allowed on a tensor that requires "
        "gradient. "
        "Please detach the tensor or clone it before proceeding.");
    return NULL;
  }
  Py_INCREF(a);
  return (PyObject *)a;
}
// ────────────────────────────────────────────
// other dunder methods
// ────────────────────────────────────────────

static PyObject *PyTensor_repr(PyObject *self) {
  // Format a string representation of your object
  return PyUnicode_FromFormat(
      ((PyTensorObject *)self)->inner->_native_obj->__repr__().c_str());
}

static Py_ssize_t PyTensor_len(PyObject *self) {
  // called as len(tensor) or tensor.__len__()
  return (Py_ssize_t)((PyTensorObject *)self)->inner->_native_obj->dims[0];
}

int PyTensor_setitem(PyObject *self, PyObject *key, PyObject *value) {
  // called as tensor[key] = value or del tensor[key] (if value is null) or
  // tensor.__setitem__(key, value);
  return -1;
}
static PyObject *PyTensor_getitem(PyTensorObject *self, PyObject *item) {
  // called as tensor[item], or tensor.__getitem__(item);
  PyTensorObject *view = PyObject_New(PyTensorObject, &PyTensorType);
  if (view == NULL)
    return NULL;

  std::vector<Slice> slices;
  Py_ssize_t start, stop, step, slicelength;
  if (PyTuple_Check(item)) {
    Py_ssize_t ndim = PyTuple_GET_SIZE(item);
    for (Py_ssize_t i = 0; i < ndim; i++) {
      Slice s;
      PyObject *item_i = PyTuple_GET_ITEM(item, i);
      if (!PySlice_Check(item_i)) {
        PyErr_SetString(PyExc_TypeError, "Index must be a slice.");

        Py_DECREF((PyObject *)view);
        return NULL;
      }

      if (PySlice_GetIndicesEx(item_i, self->inner->_native_obj->size, &start,
                               &stop, &step, &slicelength) < 0) {
        Py_DECREF((PyObject *)view);
        return NULL;
      }
      s.step = step;
      s.start = start;
      s.stop = stop;
      slices.push_back(s);
    }

    view->inner = new TensorStruct;
    view->inner->_native_obj = self->inner->_native_obj->view(slices);
    return (PyObject *)view;
  } else if (PySlice_Check(item)) {
    Slice s;
    if (PySlice_GetIndicesEx(item, self->inner->_native_obj->size, &start,
                             &stop, &step, &slicelength) < 0) {
      Py_DECREF((PyObject *)view);
      return NULL;
    }
    s.step = step;
    s.start = start;
    s.stop = stop;
    slices.push_back(s);
    view->inner = new TensorStruct;
    view->inner->_native_obj = self->inner->_native_obj->view(slices);
    return (PyObject *)view;
  } else if (PyLong_Check(item)) {
    PyErr_SetString(PyExc_TypeError, "Not Implemented.");
    return NULL;
  }
  PyErr_SetString(PyExc_TypeError, "Invalid index.");
  return NULL;
}

static PyMethodDef PyTensor_methods[] = {
    {"print", (PyCFunction)PyTensor_print, METH_NOARGS, "Print the tensor"},
    {"print_buffer", (PyCFunction)PyTensor_print_buffer, METH_NOARGS,
     "Print the Tensor Buffer"},
    {NULL}};

static PyGetSetDef PyTensor_getsets[] = {
    {"requires_grad", (getter)PyTensor_get_requires_grad,
     (setter)PyTensor_set_requires_grad,
     "Boolean flag indicating whether this tensor should track operations "
     "for "
     "gradient computation.\n"
     "When set to True, the tensor records operations to enable automatic "
     "differentiation during backpropagation.\n"
     "Defaults to False.",
     NULL},

    {NULL}};

static PyNumberMethods PyTensor_as_number = {
    .nb_add = PyTensor_add,
    .nb_subtract = PyTensor_sub,
    .nb_multiply = PyTensor_mul,
    .nb_inplace_add = (binaryfunc)PyTensor_add_inplace,
    .nb_inplace_subtract = (binaryfunc)PyTensor_sub_inplace,
    .nb_inplace_multiply = (binaryfunc)PyTensor_mul_inplace,
    .nb_true_divide = PyTensor_div,
    .nb_inplace_true_divide = (binaryfunc)PyTensor_div_inplace,
};
static PyMappingMethods PyTensor_as_mapping = {
    .mp_length = (lenfunc)PyTensor_len,
    .mp_subscript = (binaryfunc)PyTensor_getitem,
    .mp_ass_subscript = (objobjargproc)PyTensor_setitem

};

PyTypeObject PyTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "extension.tensor.Tensor",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_dealloc = (destructor)PyTensor_dealloc,
    .tp_repr = PyTensor_repr,
    .tp_as_number = &PyTensor_as_number,
    .tp_as_mapping = &PyTensor_as_mapping,
    .tp_methods = PyTensor_methods,
    .tp_getset = PyTensor_getsets,
    .tp_init = (initproc)PyTensor_init,
    .tp_new = PyTensor_new,
};

static PyObject *PyTensor_Ones(PyObject *self, PyObject *args, PyObject *kwds) {
  PyObject *dims = nullptr;
  int dtype = static_cast<int>(DType::float32);
  int requires_grad = 0;
  static const char *keywords[] = {"dims", "dtype", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ip", (char **)keywords, &dims,
                                   &dtype, &requires_grad)) {
    return NULL;
  }
  if (dims == NULL) {
    PyErr_SetString(PyExc_TypeError, "dims is a required argument.");
    return NULL;
  }
  PyTensorObject *t = PyObject_New(PyTensorObject, &PyTensorType);
  if (t == NULL) {
    return NULL;
  }

  if (PyTuple_Check(dims)) {
    Py_ssize_t ndim = PyTuple_Size(dims);
    if (ndim <= 0) {
      PyErr_SetString(PyExc_ValueError, "dims must not be empty");
      return NULL;
    }
    std::vector<int> shape;
    shape.reserve(ndim);
    for (Py_ssize_t i = 0; i < ndim; ++i) {
      PyObject *item = PyTuple_GetItem(dims, i);
      if (!PyLong_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "dims must be integers");
        return NULL;
      }
      long dim = PyLong_AsLong(item);
      if (dim <= 0) {
        PyErr_SetString(PyExc_ValueError, "dims must be positive integers");
        return NULL;
      }
      shape.push_back(static_cast<int>(dim));
    }
    t->inner = new TensorStruct;
    t->inner->_native_obj =
        Tensor::ones(shape, static_cast<DType>(dtype), requires_grad == 1);
    return (PyObject *)t;
  }
  PyErr_SetString(PyExc_ValueError,
                  "dims must be a tuple of positive integers");
  return NULL;
}

static PyObject *PyTensor_Zeros(PyObject *self, PyObject *args,
                                PyObject *kwds) {
  PyObject *dims = nullptr;
  int dtype = static_cast<int>(DType::float32);
  int requires_grad = 0;
  static const char *keywords[] = {"dims", "dtype", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ip", (char **)keywords, &dims,
                                   &dtype, &requires_grad)) {
    return NULL;
  }
  if (dims == NULL) {
    PyErr_SetString(PyExc_TypeError, "dims is a required argument.");
    return NULL;
  }
  PyTensorObject *t = PyObject_New(PyTensorObject, &PyTensorType);
  if (t == NULL) {
    return NULL;
  }

  if (PyTuple_Check(dims)) {
    Py_ssize_t ndim = PyTuple_Size(dims);
    if (ndim <= 0) {
      PyErr_SetString(PyExc_ValueError, "dims must not be empty");
      return NULL;
    }
    std::vector<int> shape;
    shape.reserve(ndim);
    for (Py_ssize_t i = 0; i < ndim; ++i) {
      PyObject *item = PyTuple_GetItem(dims, i);
      if (!PyLong_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "dims must be integers");
        return NULL;
      }
      long dim = PyLong_AsLong(item);
      if (dim <= 0) {
        PyErr_SetString(PyExc_ValueError, "dims must be positive integers");
        return NULL;
      }
      shape.push_back(static_cast<int>(dim));
    }
    t->inner = new TensorStruct;
    t->inner->_native_obj =
        Tensor::zeros(shape, static_cast<DType>(dtype), requires_grad == 1);
    return (PyObject *)t;
  }
  PyErr_SetString(PyExc_ValueError,
                  "dims must be a tuple of positive integers");
  return NULL;
}

static PyObject *PyTensor_Eye(PyObject *self, PyObject *args, PyObject *kwds) {
  int n = -1;
  int dtype = static_cast<int>(DType::float32);
  int requires_grad = 0;
  static const char *keywords[] = {"n", "dtype", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|ip", (char **)keywords, &n,
                                   &dtype, &requires_grad)) {
    return NULL;
  }
  if (n == -1) {
    PyErr_SetString(PyExc_TypeError, "Missing Argument n (order of matrix).");
    return NULL;
  }
  PyTensorObject *t = PyObject_New(PyTensorObject, &PyTensorType);
  if (t == NULL) {
    return NULL;
  }

  if (n <= 0) {
    PyErr_SetString(PyExc_ValueError, "order n must not be positive");
    return NULL;
  }
  t->inner = new TensorStruct;
  t->inner->_native_obj =
      Tensor::eye(n, static_cast<DType>(dtype), requires_grad == 1);
  return (PyObject *)t;
}

static PyMethodDef TensorModuleMethods[] = {
    {"ones", (PyCFunction)PyTensor_Ones, METH_VARARGS | METH_KEYWORDS,
     "Create a Tensor filled with ones."},
    {"zeros", (PyCFunction)PyTensor_Zeros, METH_VARARGS | METH_KEYWORDS,
     "Create a Tensor filled with zeros."},
    {"eye", (PyCFunction)PyTensor_Eye, METH_VARARGS | METH_KEYWORDS,
     "Create an identity tensor."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef tensormodule = {
    PyModuleDef_HEAD_INIT, "extension.tensor", NULL, -1, TensorModuleMethods};

PyObject *createTensorModule(PyObject *parent) {
  PyObject *tensor = PyModule_Create(&tensormodule);
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
