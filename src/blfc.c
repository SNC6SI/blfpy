//@author: SNC6SI: Shen,Chenghao <snc6si@gmail.com>

#define PY_SSIZE_T_CLEAN
#include "include.h"


# if 1
PyObject* read_info(PyObject* self, PyObject* args)
{

    HANDLE hFile;
    char* cfileName;
    BOOL bSuccess;
    Py_ssize_t len;
    DWORD statisticCnt;
    DWORD msgcnt = 0;
    VBLFileStatisticsEx statistics = { sizeof(statistics) };

    PyObject* arglist;
    PyObject* measureStartTime;
    PyObject* lastObjectTime;

    if (!PyArg_ParseTuple(args, "s#", &cfileName, &len))
        return NULL;

    hFile = BLCreateFile(cfileName, GENERIC_READ);
    if (INVALID_HANDLE_VALUE == hFile)
    {
        PyErr_SetString(PyExc_FileNotFoundError, "No such blf file or directory.");
        return NULL;
    }
    bSuccess = BLGetFileStatisticsEx(hFile, &statistics);
    if (!bSuccess)
        return NULL;
    else
    {
        statisticCnt = statistics.mObjectCount;
        measureStartTime = PyUnicode_FromFormat("%u/%u/%u/ %u:%u:%u",
            statistics.mMeasurementStartTime.wYear,
            statistics.mMeasurementStartTime.wMonth,
            statistics.mMeasurementStartTime.wDay,
            statistics.mMeasurementStartTime.wHour,
            statistics.mMeasurementStartTime.wMinute,
            statistics.mMeasurementStartTime.wSecond);
        lastObjectTime = PyUnicode_FromFormat("%u/%u/%u/ %u:%u:%u",
            statistics.mLastObjectTime.wYear,
            statistics.mLastObjectTime.wMonth,
            statistics.mLastObjectTime.wDay,
            statistics.mLastObjectTime.wHour,
            statistics.mLastObjectTime.wMinute,
            statistics.mLastObjectTime.wSecond);

        arglist = Py_BuildValue("{s:k,s:B,s:B,s:B,s:B,s:K,s:K,s:k,s:k,s:O,s:O}",
            "mStatisticsSize", statistics.mStatisticsSize,
            "mApplicationID", statistics.mApplicationID,
            "mApplicationMajor", statistics.mApplicationMajor,
            "mApplicationMinor", statistics.mApplicationMinor,
            "mApplicationBuild", statistics.mApplicationBuild,
            "mFileSize", statistics.mFileSize,
            "mUncompressedFileSize", statistics.mUncompressedFileSize,
            "mObjectCount", statistics.mObjectCount,
            "mObjectsRead", statistics.mObjectsRead,
            "mMeasurementStartTime", measureStartTime,
            "mLastObjectTime", lastObjectTime);

        if (!BLCloseHandle(hFile))
        {
            return NULL;
        }
        return arglist;
    }
}
#endif

PyObject* read_data(PyObject* self, PyObject* args)
{

    char* cfileName;
    HANDLE hFile;
    BOOL bSuccess;
    Py_ssize_t len;
    DWORD statisticCnt = 0;
    DWORD msgcnt = 0;
    VBLFileStatisticsEx statistics = { sizeof(statistics) };
    VBLObjectHeaderBase base;
    VBLCANMessage message;
    VBLCANMessage2 message2;
    size_t i;

    unsigned char * u8_candata, *u8_tmp;
//     long long * ll64_candata, * ll64_tmp;
    unsigned short * u16_canchannel, * u16_tmp;
    unsigned long * u32_canmsgid, * u32_tmp;
    double * f64_cantime, * f64_tmp;

    //double* candata, * canmsgid, * canchannel, * cantime;
    PyObject * arglist;
    PyObject * L_candata, * L_candata_;
    PyObject * L_canmsgid;
    PyObject * L_canchannel;
    PyObject * L_cantime;
    npy_intp dimcandata, dimcanmsg;
    npy_intp d[2];
    PyArray_Dims bytedim;
    
    bytedim.ptr = d;
    bytedim.len = 2;

    if (!PyArg_ParseTuple(args, "s#", &cfileName, &len))
        return NULL;

    hFile = BLCreateFile(cfileName, GENERIC_READ);
    if (INVALID_HANDLE_VALUE == hFile)
    {
        PyErr_SetString(PyExc_FileNotFoundError, "No such blf file or directory.");
        return NULL;
    }
    bSuccess = BLGetFileStatisticsEx(hFile, &statistics);
    if (!bSuccess)
        return NULL;
    else
    statisticCnt = statistics.mObjectCount;

    // allocate memroy
    u8_candata = (unsigned char *)PyMem_Malloc(((size_t)(statisticCnt)) * (sizeof(unsigned char)) * 8);
    u32_canmsgid = (unsigned long *)PyMem_Malloc(((size_t)(statisticCnt)) * sizeof(unsigned long));
    u16_canchannel = (unsigned short *)PyMem_Malloc(((size_t)(statisticCnt)) * sizeof(unsigned short));
    f64_cantime = (double *)PyMem_Malloc(((size_t)(statisticCnt)) * sizeof(double));

    // =======================================================================
    // BEGIN read data
    // =======================================================================
    while (bSuccess && BLPeekObject(hFile, &base))
    {
        switch (base.mObjectType)
        {
        case BL_OBJ_TYPE_CAN_MESSAGE:
            message.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &message.mHeader.mBase, sizeof(message));
            if (bSuccess) {
                for (i = 0; i < 8; i++)
                {
                    * (u8_candata + (((size_t)msgcnt) << 3) + i) = (unsigned char)message.mData[i];
                }

                *(u32_canmsgid + msgcnt) = message.mID;

                *(u16_canchannel + msgcnt) = message.mChannel;

                if (message.mHeader.mObjectFlags == BL_OBJ_FLAG_TIME_ONE_NANS)
                {
                    *(f64_cantime + msgcnt) = ((double)message.mHeader.mObjectTimeStamp) / 1000000000;
                }
                else
                {
                    *(f64_cantime + msgcnt) = ((double)message.mHeader.mObjectTimeStamp) / 100000;
                }
                BLFreeObject(hFile, &message.mHeader.mBase);
                msgcnt++;
            }
            break;
        case BL_OBJ_TYPE_CAN_MESSAGE2:
            message2.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &message2.mHeader.mBase, sizeof(message2));
            if (bSuccess) {
                for (i = 0; i < 8; i++)
                {
                    *(u8_candata + (((size_t)msgcnt) << 3) + i) = (unsigned char)message2.mData[i];
                }  

                *(u32_canmsgid + msgcnt) = message2.mID;

                *(u16_canchannel + msgcnt) = message2.mChannel;

                if (message2.mHeader.mObjectFlags == BL_OBJ_FLAG_TIME_ONE_NANS)
                {
                    *(f64_cantime + msgcnt) = ((double)message2.mHeader.mObjectTimeStamp) / 1000000000;
                }
                else
                {
                    *(f64_cantime + msgcnt) = ((double)message2.mHeader.mObjectTimeStamp) / 100000;
                }
                BLFreeObject(hFile, &message2.mHeader.mBase);
                msgcnt++;
            }
            break;
        case BL_OBJ_TYPE_ENV_INTEGER:
        case BL_OBJ_TYPE_ENV_DOUBLE:
        case BL_OBJ_TYPE_ENV_STRING:
        case BL_OBJ_TYPE_ENV_DATA:
        case BL_OBJ_TYPE_ETHERNET_FRAME:
        case BL_OBJ_TYPE_APP_TEXT:
        default:
            /* skip all other objects */
            bSuccess = BLSkipObject(hFile, &base);
            break;
        };
    }

    if (!BLCloseHandle(hFile))
    {
        return NULL;
    }

    // =======================================================================
    // END read data
    // =======================================================================

    // reallocate memory if needed
    if (statisticCnt!= msgcnt)
    {
        u8_tmp = (unsigned char *)PyMem_Realloc(u8_candata, ((size_t)(msgcnt)) * (sizeof(unsigned char)) * 8);
        if (u8_tmp != NULL) u8_candata = u8_tmp;
        u32_tmp = (unsigned long *)PyMem_Realloc(u32_canmsgid, ((size_t)(msgcnt)) * sizeof(unsigned long));
        if (u32_tmp != NULL) u32_canmsgid = u32_tmp;
        u16_tmp = (unsigned short *)PyMem_Realloc(u16_canchannel, ((size_t)(msgcnt)) * sizeof(unsigned short));
        if (u16_tmp != NULL) u16_canchannel = u16_tmp;
        f64_tmp = (double*)PyMem_Realloc(f64_cantime, ((size_t)(msgcnt)) * sizeof(double));
        if (f64_tmp != NULL) f64_cantime = f64_tmp;
    }

    // create numpy ndarray
    // set owndata is important for memory management (free memory) in python
    dimcandata = ((npy_intp)msgcnt) << 3;
    dimcanmsg = ((npy_intp)msgcnt);
    L_candata_    = PyArray_SimpleNewFromData(1, &dimcandata, NPY_UINT8, u8_candata);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_candata_, NPY_ARRAY_OWNDATA);
    bytedim.ptr[0] = msgcnt;
    bytedim.ptr[1] = 8;
    L_candata = PyArray_Newshape((PyArrayObject *)L_candata_, &bytedim, NPY_CORDER);
    // L_candata = PyArray_Newshape((PyArrayObject *)L_candata_, &bytedim, NPY_FORTRANORDER);
    Py_DECREF(L_candata_);
    
    L_canmsgid   = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_ULONG, u32_canmsgid);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_canmsgid, NPY_ARRAY_OWNDATA);
    
    L_canchannel = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_USHORT, u16_canchannel);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_canchannel, NPY_ARRAY_OWNDATA);
    
    L_cantime    = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_DOUBLE, f64_cantime);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_cantime, NPY_ARRAY_OWNDATA);
    
    arglist = Py_BuildValue("(NNNN)", L_candata, L_canmsgid, L_canchannel, L_cantime);

    return arglist;
}
    



static PyMethodDef blfcMethods[] = {
    {"read_info", (PyCFunction)read_info, METH_VARARGS, 0},
    {"read_data", (PyCFunction)read_data, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef blfcmodule = {
    PyModuleDef_HEAD_INIT,
    "blfc",
    "Shen, Chenghao <snc6si@gmail.com>", 
    -1,
    blfcMethods
};


PyMODINIT_FUNC
PyInit_blfc(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&blfcmodule);
}
