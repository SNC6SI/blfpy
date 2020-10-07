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
static VBLObjectHeaderContainer container;
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


static uint32_t objectCounts[4] = {0, 0, 0, 0};
static uint8_t compressedDataWrite[BL_CHUNK];
static uint8_t unCompressedDataWrite[BL_CHUNK];
static uint32_t thisSize = 0;
static uint8_t paddingBytesWrite[3] = {0, 0, 0};
static uint32_t paddingSize = 0;

static uint8_t i;

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
            fread(&container, BL_HEADER_CONTAINER_SIZE, 1, fp);
            //
            compressedSize = container.mBase.mObjectSize - 
                             BL_HEADER_CONTAINER_SIZE;
            compressedData = PyMem_Malloc(compressedSize);
            fread(compressedData, compressedSize, 1, fp);
            if(paddingBytes > 0){
                fseek(fp, paddingBytes, SEEK_CUR);
            }
            //
            unCompressedSize = container.mDeflatebuffersize + restSize;
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



void loggInit(void){
    logg.mSignature = BL_LOGG_SIGNATURE;
    logg.mHeaderSize = BL_LOGG_SIZE;
    logg.mCRC = 0;
    logg.mApplicationID = BL_APPID_BLFLOAD;
    logg.mdwCompression = 0;
    logg.mApplicationMajor = BL_Major;
    logg.mApplicationMinor = BL_Minor;
    logg.mApplicationBuild = BL_AppBuild;
    //
    logg.mFileSize = BL_LOGG_SIZE;
    logg.mUncompressedFileSize = BL_LOGG_SIZE;
    logg.mObjectCount = 0;
}


void messageInit(void){
    message.mHeader.mBase.mSignature = BL_OBJ_SIGNATURE;
    message.mHeader.mBase.mHeaderSize = BL_HEADER_SIZE;
    message.mHeader.mBase.mHeaderVersion = 1;
    message.mHeader.mBase.mObjectSize = BL_MESSAGE_SIZE;
    message.mHeader.mBase.mObjectType = BL_OBJ_TYPE_CAN_MESSAGE;
    //
    message.mHeader.mObjectFlags = BL_OBJ_FLAG_TIME_ONE_NANS;
    //
    message.mDLC = 8;
}


void containerInit(void){
    container.mBase.mSignature = BL_OBJ_SIGNATURE;
    container.mBase.mHeaderSize = BL_HEADER_BASE_SIZE;
    container.mBase.mHeaderVersion = 1;
    container.mBase.mObjectType = BL_OBJ_TYPE_LOG_CONTAINER;
    //
    container.mCompressedflag = 2;
}


void blfWriteInit(void){
    filename = NULL;
    fp = NULL;
    //
    memset(&logg, 0, BL_LOGG_SIZE);
    loggInit();
    //
    memset(&message, 0, BL_MESSAGE_SIZE);
    messageInit();
    //
    memset(&container, 0, BL_HEADER_CONTAINER_SIZE);
    containerInit();
    //
    contFlag = 0;
    rcnt = 0;
    objectCounts[0] = 0;
    objectCounts[1] = 0;
    objectCounts[2] = 0;
    objectCounts[3] = 0;
    //
    compressedSize = 0;
    unCompressedSize = 0;
    thisSize = 0;
    restSize = 0;
    paddingSize = 0;
}


void blfWriteObjectInternal(void){
    compressedSize = BL_CHUNK;
    compress(compressedDataWrite, &compressedSize, unCompressedDataWrite, unCompressedSize);
    container.mBase.mObjectSize = BL_HEADER_CONTAINER_SIZE + compressedSize;
    container.mDeflatebuffersize = unCompressedSize;
    //
    fwrite(&container, BL_HEADER_CONTAINER_SIZE, 1, fp);
    fwrite(compressedDataWrite, compressedSize, 1, fp);
    paddingSize = compressedSize & 3;
    if(paddingSize > 0){
        fwrite(paddingBytesWrite, paddingSize, 1, fp);
    }
    //
    logg.mFileSize += BL_HEADER_CONTAINER_SIZE + compressedSize + paddingSize;
    logg.mUncompressedFileSize += BL_HEADER_CONTAINER_SIZE + unCompressedSize;
    //
    unCompressedSize = 0;
}


uint8_t blfWriteObject(void){
    while(rcnt < objectCounts[0]){
        if(contFlag){
            memcpy(((uint8_t *)unCompressedDataWrite) + unCompressedSize, ((uint8_t *)&message) + thisSize, restSize);
            unCompressedSize += restSize;
            contFlag = 0;
            rcnt++;
        }else{
            message.mHeader.mObjectTimeStamp = (uint64_t)((*(f64_cantime + rcnt))*1000000000);
            message.mChannel = (uint16_t)(*(u16_canchannel + rcnt));
            message.mID = (uint32_t)(*(u32_canmsgid + rcnt));
            for(i=0;i<8;i++){
                message.mData[i] = (uint8_t)(*(u8_candata + rcnt*8 +i));
            }
            if(unCompressedSize + BL_MESSAGE_SIZE <= BL_CHUNK){
                memcpy(((uint8_t *)unCompressedDataWrite) + unCompressedSize, &message, BL_MESSAGE_SIZE);
                unCompressedSize += BL_MESSAGE_SIZE;
                rcnt++;
            }else{
                thisSize = BL_CHUNK - unCompressedSize;
                restSize = unCompressedSize + BL_MESSAGE_SIZE - BL_CHUNK;
                if(thisSize!=0){
                    memcpy(((uint8_t *)unCompressedDataWrite) + unCompressedSize, &message, thisSize);
                    unCompressedSize += thisSize;
                }
                contFlag = 1;
            }
        }
        if(contFlag){
            blfWriteObjectInternal();
        }
    }
    //
    if(unCompressedSize > 0){
        blfWriteObjectInternal();
    }
}

PyObject * write_data(PyObject *self, PyObject *args) {
    Py_ssize_t len;
    PyObject * O_data, * O_id, * O_channel, * O_time;
    npy_intp* pdim_data, * pdim_id, * pdim_channel, * pdim_time;

    blfWriteInit();

    if (!PyArg_ParseTuple(args, "s#(OOOO)", &filename, &len, &O_data, &O_id, &O_channel, &O_time))
        return NULL;
    pdim_data = PyArray_SHAPE(O_data);
    pdim_id = PyArray_SHAPE(O_id);
    pdim_channel = PyArray_SHAPE(O_channel);
    pdim_time = PyArray_SHAPE(O_time);
    objectCounts[0] = pdim_data[0];
    objectCounts[1] = pdim_id[0];
    objectCounts[2] = pdim_channel[0];
    objectCounts[3] = pdim_time[0];

    u8_candata = (npy_ubyte*)PyArray_DATA((PyArrayObject*)O_data);
    u32_canmsgid = (npy_uint32*)PyArray_DATA((PyArrayObject*)O_id);
    u16_canchannel = (npy_uint16*)PyArray_DATA((PyArrayObject*)O_channel);
    f64_cantime = (npy_double*)PyArray_DATA((PyArrayObject*)O_time);

    fp = fopen(filename, "wb");
    if(fp==NULL){
        PyErr_SetString(PyExc_FileNotFoundError, strerror(errno));
        return NULL;
    }
    fseek(fp, BL_LOGG_SIZE, SEEK_SET);
    blfWriteObject();
    logg.mObjectCount = rcnt;
    fseek(fp, 0, SEEK_SET);
    fwrite(&logg, BL_LOGG_SIZE, 1, fp);
    fclose(fp);
    Py_RETURN_NONE;
}



static PyMethodDef blfcMethods[] = {
    {"read_info", (PyCFunction)read_info, METH_VARARGS, 0},
    {"read_data", (PyCFunction)read_data, METH_VARARGS, 0},
    {"write_data", (PyCFunction)write_data, METH_VARARGS, 0},
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
