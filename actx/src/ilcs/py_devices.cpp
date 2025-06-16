#include "ilcs/py_devices.h"
#include "device_type.h"

PyObject *createDevicesModule(PyObject *parent) {
  PyObject *devices = PyModule_New("extension.devices");
  if (devices == NULL) {
    Py_DECREF(parent);
    return NULL;
  }
  PyModule_AddIntConstant(devices, "CPU", static_cast<int>(DeviceType::CPU));
  PyModule_AddIntConstant(devices, "MPS", static_cast<int>(DeviceType::MPS));
  PyModule_AddIntConstant(devices, "WEBGPU",
                          static_cast<int>(DeviceType::WEBGPU));
  return devices;
}
