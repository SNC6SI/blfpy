#include "include.h"


# if 1
PyObject* readFileInfos(PyObject* self, PyObject* args)
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

PyObject* readData(PyObject* self, PyObject* args)
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
    PyObject* arglist;
    PyObject * L_candata;
    PyObject * L_canmsgid;
    PyObject * L_canchannel;
    PyObject * L_cantime;
    npy_intp dimcandata, dimcanmsg;

    if (!PyArg_ParseTuple(args, "s#", &cfileName, &len))
        return NULL;

    hFile = BLCreateFile(cfileName, GENERIC_READ);
    if (INVALID_HANDLE_VALUE == hFile)
    {
        return NULL;
    }
    bSuccess = BLGetFileStatisticsEx(hFile, &statistics);
    if (!bSuccess)
        return NULL;
    else
    statisticCnt = statistics.mObjectCount;

    // allocate memroy
    u8_candata = (unsigned char *)malloc(((size_t)(statisticCnt)) * (sizeof(unsigned char)) * 8);
    u32_canmsgid = (unsigned long *)malloc(((size_t)(statisticCnt)) * sizeof(unsigned long));
    u16_canchannel = (unsigned short *)malloc(((size_t)(statisticCnt)) * sizeof(unsigned short));
    f64_cantime = (double *)malloc(((size_t)(statisticCnt)) * sizeof(double));

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
        u8_tmp = (unsigned char *)realloc(u8_candata, ((size_t)(msgcnt)) * (sizeof(unsigned char)) * 8);
        if (u8_tmp != NULL) u8_candata = u8_tmp;
        u32_tmp = (unsigned long *)realloc(u32_canmsgid, ((size_t)(msgcnt)) * sizeof(unsigned long));
        if (u32_tmp != NULL) u32_canmsgid = u32_tmp;
        u16_tmp = (unsigned short *)realloc(u16_canchannel, ((size_t)(msgcnt)) * sizeof(unsigned short));
        if (u16_tmp != NULL) u16_canchannel = u16_tmp;
        f64_tmp = (double*)realloc(f64_cantime, ((size_t)(msgcnt)) * sizeof(double));
        if (f64_tmp != NULL) f64_cantime = f64_tmp;
    }

    // create numpy ndarray
    // set owndata is important for memory management (free memory) in python
    dimcandata = ((npy_intp)msgcnt) << 3;
    dimcanmsg = ((npy_intp)msgcnt);
    L_candata    = PyArray_SimpleNewFromData(1, &dimcandata, NPY_UINT8, u8_candata);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_candata, NPY_ARRAY_OWNDATA);
    
    L_canmsgid   = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_ULONG, u32_canmsgid);
    PyArray_ENABLEFLAGS((PyArrayObject*)L_canmsgid, NPY_ARRAY_OWNDATA);
    
    L_canchannel = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_USHORT, u16_canchannel);
    PyArray_ENABLEFLAGS((PyArrayObject*)L_canchannel, NPY_ARRAY_OWNDATA);
    
    L_cantime    = PyArray_SimpleNewFromData(1, &dimcanmsg, NPY_DOUBLE, f64_cantime);
    PyArray_ENABLEFLAGS((PyArrayObject*)L_cantime, NPY_ARRAY_OWNDATA);
    
    arglist = Py_BuildValue("(NNNN)", L_candata, L_canmsgid, L_canchannel, L_cantime);

    return arglist;
}
    



static PyMethodDef blfpyMethods[] = {
    {"readFileInfo", (PyCFunction)readFileInfos, METH_VARARGS, 0},
    {"readFileData", (PyCFunction)readData, METH_VARARGS, 0},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef blfpymodule = {
    PyModuleDef_HEAD_INIT,
    "blfpy",   /* name of module */
    "contact author for more info: snc6si@gmail.com", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    blfpyMethods
};


PyMODINIT_FUNC
PyInit_blfpy(void)
{
    import_array();
    import_umath();
    return PyModule_Create(&blfpymodule);
}
