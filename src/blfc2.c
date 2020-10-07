//@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>

#define PY_SSIZE_T_CLEAN
#include "include.h"


static char *filename = NULL;
static FILE *fp = NULL;
static long int filelen = 0;
static long int midx = 0;
static long int lidx = 0;
static LOGG_t logg;
static VBLObjectHeaderBase Base;
static VBLObjectHeaderContainer Container;
static VBLCANMessage message;
static VBLFileStatisticsEx pStatistics;
static uint8_t peekFlag = 1;
static uint8_t contFlag = 0;
static uint32_t rcnt = 0;

static uint8_t *compressedData = NULL;
static uint32_t compressedSize = 0;
static uint8_t *unCompressedData = NULL;
static uint32_t unCompressedSize = 0;
static uint8_t *restData = NULL;
static uint32_t restSize = 0;

extern int errno;

unsigned char * u8_candata, *u8_tmp;
unsigned short * u16_canchannel, * u16_tmp;
unsigned long * u32_canmsgid, * u32_tmp;
double * f64_cantime, * f64_tmp;


void blfInit(void){
    filename = NULL;
    fp = NULL;
    filelen = 0;
    midx = 0;
    lidx = 0;
    peekFlag = 1;
    contFlag = 0;
    rcnt = 0;
    compressedData = NULL;
    compressedSize = 0;
    unCompressedData = NULL;
    unCompressedSize = 0;
    restData = NULL;
    restSize = 0;
}

void blfStatisticsFromLogg(void){
    pStatistics.mApplicationID = logg.mApplicationID;
    pStatistics.mApplicationMajor = logg.mApplicationMajor;
    pStatistics.mApplicationMinor = logg.mApplicationMinor;
    pStatistics.mApplicationBuild = logg.mApplicationBuild;
    pStatistics.mFileSize = logg.mFileSize;
    pStatistics.mUncompressedFileSize = logg.mUncompressedFileSize;
    pStatistics.mObjectCount = logg.mObjectCount;
    pStatistics.mMeasurementStartTime = logg.mMeasurementStartTime;
    pStatistics.mLastObjectTime = logg.mLastObjectTime;
}

int memUncompress(uint8_t  *next_out,
     uint32_t  avail_out,
     uint8_t  *next_in,
     uint32_t  avail_in,
     uint32_t *total_out_ptr)
{
  int zres;
  z_stream stream;

  stream.next_in = next_in;
  stream.avail_in = avail_in;
  stream.next_out = next_out;
  stream.avail_out = avail_out;
  stream.total_out = 0;
  stream.state = NULL;
  stream.zalloc = NULL;
  stream.zfree = NULL;

  zres = inflateInit_(&stream, ZLIB_VERSION, sizeof(stream));
  if(zres == Z_OK) zres = inflate(&stream, Z_FINISH);
  if(zres == Z_STREAM_END) zres = Z_OK;
  if(zres == Z_OK) {
    inflateEnd(&stream);
    if(total_out_ptr != NULL) {
      *total_out_ptr = stream.total_out;
    }
  }
  return zres == Z_OK;
}

uint8_t blfPeekObject(){
    uint8_t success = 1;
    uint32_t paddingBytes;
    long int midx_;
    while(peekFlag){
        midx_ = midx + BL_HEADER_BASE_SIZE;
        if(midx_ >= filelen){
            success = 0;
            return success;
        }
        //
        fread(&Base, BL_HEADER_BASE_SIZE, 1, fp);
        //
        paddingBytes = Base.mObjectSize & 3;
        midx_ = midx + Base.mObjectSize + paddingBytes;
        if(midx_ > filelen){
            success = 0;
            return success;
        }
        //
        fseek(fp, -BL_HEADER_BASE_SIZE, SEEK_CUR);
        //
        if(Base.mObjectType == BL_OBJ_TYPE_LOG_CONTAINER){
            if(contFlag){
                restSize = unCompressedSize - lidx;
                if(restSize > 0){
                    restData = PyMem_Malloc(restSize);
                    memcpy(restData, unCompressedData + lidx, restSize);
                }
            }
            PyMem_Free(unCompressedData);
            fread(&Container, BL_HEADER_CONTAINER_SIZE, 1, fp);
            //
            compressedSize = Container.base.mObjectSize - 
                             BL_HEADER_CONTAINER_SIZE;
            compressedData = PyMem_Malloc(compressedSize);
            fread(compressedData, compressedSize, 1, fp);
            if(paddingBytes > 0){
                fseek(fp, paddingBytes, SEEK_CUR);
            }
            //
            unCompressedSize = Container.deflatebuffersize + restSize;
            unCompressedData = PyMem_Malloc(unCompressedSize);
            memUncompress(unCompressedData + restSize,
                          unCompressedSize - restSize,
                          compressedData,
                          compressedSize,
                          0);
            //
            PyMem_Free(compressedData);
            if(restSize > 0){
                memcpy(unCompressedData, restData, restSize);
                PyMem_Free(restData);
            }
            //
            restSize = 0;
            lidx = 0;
            peekFlag = 0;
        }else{
            fseek(fp, Base.mObjectSize + paddingBytes, SEEK_CUR);
        }
        midx = midx_;
    }
    return success;
}

uint8_t blfReadObjectSecure(){
    long int lidx_;
    int i;
    uint32_t paddingBytes;
    lidx_ = lidx + BL_HEADER_BASE_SIZE;
    if(lidx_ >= unCompressedSize){
        peekFlag = 1;
        contFlag = 1;
        return 0;
    }
    //
    memcpy(&Base, unCompressedData+lidx, BL_HEADER_BASE_SIZE);
    //
    paddingBytes = Base.mObjectSize & 3;
    lidx_ = lidx + Base.mObjectSize + paddingBytes;
    if(lidx_ > unCompressedSize){
        peekFlag = 1;
        contFlag = 1;
        return 0;
    }
    //
    contFlag = 0;
    switch (Base.mObjectType){
        case BL_OBJ_TYPE_CAN_MESSAGE:
        case BL_OBJ_TYPE_CAN_MESSAGE2:
            memcpy(&message, unCompressedData+lidx, BL_MESSAGE_SIZE);
            for(i=0;i<8;i++) *(u8_candata + rcnt*8 + i) = message.mData[i];
            *(u32_canmsgid + rcnt) = message.mID;
            *(u16_canchannel + rcnt) = message.mChannel;
            if(message.mHeader.mObjectFlags==BL_OBJ_FLAG_TIME_ONE_NANS)
            *(f64_cantime + rcnt) = 
                ((double)message.mHeader.mObjectTimeStamp)/1000000000;
            else
            *(f64_cantime + rcnt) = 
                ((double)message.mHeader.mObjectTimeStamp)/100000;
            rcnt++;
            break;
        default:
            break;
    }
    lidx = lidx_;
    return 1;
}


PyObject* read_info(PyObject* self, PyObject* args)
{
    Py_ssize_t len;

    PyObject* arglist;
    PyObject* measureStartTime;
    PyObject* lastObjectTime;

    blfInit();

    if (!PyArg_ParseTuple(args, "s#", &filename, &len))
        return NULL;

    fp = fopen(filename, "rb");
    if(fp==NULL){
        PyErr_SetString(PyExc_FileNotFoundError, strerror(errno));
        return NULL;
    }

    fread(&logg, sizeof(LOGG_t), 1, fp);
    blfStatisticsFromLogg();

    measureStartTime = PyUnicode_FromFormat("%u/%u/%u/ %u:%u:%u",
        logg.mMeasurementStartTime.wYear,
        logg.mMeasurementStartTime.wMonth,
        logg.mMeasurementStartTime.wDay,
        logg.mMeasurementStartTime.wHour,
        logg.mMeasurementStartTime.wMinute,
        logg.mMeasurementStartTime.wSecond);
    lastObjectTime = PyUnicode_FromFormat("%u/%u/%u/ %u:%u:%u",
        logg.mLastObjectTime.wYear,
        logg.mLastObjectTime.wMonth,
        logg.mLastObjectTime.wDay,
        logg.mLastObjectTime.wHour,
        logg.mLastObjectTime.wMinute,
        logg.mLastObjectTime.wSecond);

    arglist = Py_BuildValue("{s:B,s:B,s:B,s:B,s:K,s:K,s:k,s:O,s:O}",
        "mApplicationID", logg.mApplicationID,
        "mApplicationMajor", logg.mApplicationMajor,
        "mApplicationMinor", logg.mApplicationMinor,
        "mApplicationBuild", logg.mApplicationBuild,
        "mFileSize", logg.mFileSize,
        "mUncompressedFileSize", logg.mUncompressedFileSize,
        "mObjectCount", logg.mObjectCount,
        "mMeasurementStartTime", measureStartTime,
        "mLastObjectTime", lastObjectTime);

    fclose(fp);
    return arglist;

}


PyObject* read_data(PyObject* self, PyObject* args)
{
    Py_ssize_t len;
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

    blfInit();

    if (!PyArg_ParseTuple(args, "s#", &filename, &len))
        return NULL;

    fp = fopen(filename, "rb");
    if(fp==NULL){
        PyErr_SetString(PyExc_FileNotFoundError, strerror(errno));
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filelen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    fread(&logg, sizeof(LOGG_t), 1, fp);
    blfStatisticsFromLogg();

    // allocate memroy
    u8_candata = (unsigned char *)PyMem_Malloc(((size_t)(logg.mObjectCount)) * (sizeof(unsigned char)) * 8);
    u32_canmsgid = (unsigned long *)PyMem_Malloc(((size_t)(logg.mObjectCount)) * sizeof(unsigned long));
    u16_canchannel = (unsigned short *)PyMem_Malloc(((size_t)(logg.mObjectCount)) * sizeof(unsigned short));
    f64_cantime = (double *)PyMem_Malloc(((size_t)(logg.mObjectCount)) * sizeof(double));

    // =======================================================================
    // BEGIN read data
    // =======================================================================
   while(blfPeekObject())
        blfReadObjectSecure();

    fclose(fp);
    // =======================================================================
    // END read data
    // =======================================================================

    // reallocate memory if needed
    if (logg.mObjectCount!= rcnt)
    {
        u8_tmp = (unsigned char *)PyMem_Realloc(u8_candata, ((size_t)(rcnt)) * (sizeof(unsigned char)) * 8);
        if (u8_tmp != NULL) u8_candata = u8_tmp;
        u32_tmp = (unsigned long *)PyMem_Realloc(u32_canmsgid, ((size_t)(rcnt)) * sizeof(unsigned long));
        if (u32_tmp != NULL) u32_canmsgid = u32_tmp;
        u16_tmp = (unsigned short *)PyMem_Realloc(u16_canchannel, ((size_t)(rcnt)) * sizeof(unsigned short));
        if (u16_tmp != NULL) u16_canchannel = u16_tmp;
        f64_tmp = (double*)PyMem_Realloc(f64_cantime, ((size_t)(rcnt)) * sizeof(double));
        if (f64_tmp != NULL) f64_cantime = f64_tmp;
    }

    // create numpy ndarray
    // set owndata is important for memory management (free memory) in python
    dimcandata = ((npy_intp)rcnt) << 3;
    dimcanmsg = ((npy_intp)rcnt);
    L_candata_    = PyArray_SimpleNewFromData(1, &dimcandata, NPY_UINT8, u8_candata);
    PyArray_ENABLEFLAGS((PyArrayObject *)L_candata_, NPY_ARRAY_OWNDATA);
    bytedim.ptr[0] = rcnt;
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

#if 0
PyObject * write_data(PyObject *self, PyObject *args) {
    HANDLE hFile;
    SYSTEMTIME record_time;
    BOOL bSuccess;
    char* cfileName;
    Py_ssize_t len;
    PyObject * rec_time_input, * rec_time_item;
    PyObject * O_data, * O_id, * O_channel, * O_time;
    npy_intp* pdim_data, * pdim_id, * pdim_channel, * pdim_time;
    npy_int len_message[4], len_min, i,j;
    VBLCANMessage2 message;
    npy_ubyte* data;
    npy_uint32* id;
    npy_uint16* channel;
    npy_double* time;
    char a;

    if (!PyArg_ParseTuple(args, "s#(OOOO)O", &cfileName, &len, &O_data, &O_id, &O_channel, &O_time, &rec_time_input))
        return NULL;
    pdim_data = PyArray_SHAPE(O_data);
    pdim_id = PyArray_SHAPE(O_id);
    pdim_channel = PyArray_SHAPE(O_channel);
    pdim_time = PyArray_SHAPE(O_time);
    len_message[0] = pdim_data[0];
    len_message[1] = pdim_id[0];
    len_message[2] = pdim_channel[0];
    len_message[3] = pdim_time[0];
    len_min = len_message[0];
    for (i = 1; i < 4; i++) {
        if (len_message[i] < len_min) {
            len_min = len_message[i];
        }
    }

    data = (npy_ubyte*)PyArray_DATA((PyArrayObject*)O_data);
    id = (npy_uint32*)PyArray_DATA((PyArrayObject*)O_id);
    channel = (npy_uint16*)PyArray_DATA((PyArrayObject*)O_channel);
    time = (npy_double*)PyArray_DATA((PyArrayObject*)O_time);

    hFile = BLCreateFile(cfileName, GENERIC_WRITE);
    if (INVALID_HANDLE_VALUE == hFile)
    {
        return -1;
    }
    bSuccess = BLSetApplication(hFile, BL_APPID_UNKNOWN, 1, 0, 0);
    //GetSystemTime(&systemTime);
    rec_time_item = PyDict_GetItemString(rec_time_input, "year");
    record_time.wYear = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "month");
    record_time.wMonth = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "weekday");
    record_time.wDayOfWeek = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "day");
    record_time.wDay = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "hour");
    record_time.wHour = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "minute");
    record_time.wMinute = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "second");
    record_time.wSecond = (WORD)PyLong_AsLong(rec_time_item);

    rec_time_item = PyDict_GetItemString(rec_time_input, "millisecond");
    record_time.wMilliseconds = (WORD)PyLong_AsLong(rec_time_item);

    bSuccess = bSuccess && BLSetMeasurementStartTime(hFile, &record_time);
    bSuccess = bSuccess && BLSetWriteOptions(hFile, 6, 0);
    if (bSuccess) {
        memset(&message, 0, sizeof(VBLCANMessage2));

        message.mHeader.mBase.mSignature = BL_OBJ_SIGNATURE;
        message.mHeader.mBase.mHeaderSize = sizeof(message.mHeader);
        message.mHeader.mBase.mHeaderVersion = 1;
        message.mHeader.mBase.mObjectSize = sizeof(VBLCANMessage);
        message.mHeader.mBase.mObjectType = BL_OBJ_TYPE_CAN_MESSAGE;
        message.mHeader.mObjectFlags = BL_OBJ_FLAG_TIME_ONE_NANS;
    }
    for (i = 0; i < len_min; i++) {


        message.mHeader.mObjectTimeStamp = (npy_ulonglong)((*(time+i)) * 1000000000);
        message.mChannel = *(channel+i);
        message.mFlags = CAN_MSG_FLAGS(0, 0);
        message.mDLC = 8;
        message.mID = *(id+i);
        for (j = 0; j < 8; j++)
        {
            message.mData[j] = *(data + (i << 3) + j);
        }

        bSuccess = BLWriteObject(hFile, &message.mHeader.mBase);
    }
    if (!BLCloseHandle(hFile))
    {
        return -1;
    }
    Py_RETURN_NONE;
}
#endif


static PyMethodDef blfcMethods[] = {
    {"read_info", (PyCFunction)read_info, METH_VARARGS, 0},
    {"read_data", (PyCFunction)read_data, METH_VARARGS, 0},
    //{"write_data", (PyCFunction)write_data, METH_VARARGS, 0},
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
