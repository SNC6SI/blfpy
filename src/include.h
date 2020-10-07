//#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define Py_LIMITED_API 0x03060000
#include <python.h>
#include <stdint.h>
#if defined ( _MSC_VER )
    #include <Windows.h>
#endif
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/halffloat.h>
#include "zlib/zlib.h"
#include "blfc2.h"