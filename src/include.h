#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define Py_LIMITED_API 0x03060000
#include "Python.h"
#include <stdint.h>
#ifdef _MSC_VER 
    #include <Windows.h>
#else
    struct _typeobject {};
#endif
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/halffloat.h>
#include "zlib/zlib.h"
#include "blfc2.h"