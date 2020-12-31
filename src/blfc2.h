#define BL_OBJ_TYPE_CAN_MESSAGE 1
#define BL_OBJ_TYPE_LOG_CONTAINER 10
#define BL_OBJ_TYPE_CAN_MESSAGE2 86

#define BL_OBJ_TYPE_CAN_ERROR_EXT 73

#define BL_OBJ_FLAG_TIME_TEN_MICS 0x00000001 /* 10 micro second timestamp */
#define BL_OBJ_FLAG_TIME_ONE_NANS 0x00000002 /* 1 nano second timestamp */

#define BL_HEADER_BASE_SIZE 16
#define BL_HEADER_SIZE 32
#define BL_HEADER_CONTAINER_SIZE 32
#define BL_MESSAGE_SIZE 48
#define BL_LOGG_SIZE 144
#define BL_ERROREXT_SIZE 64

#define BL_OBJ_SIGNATURE  0x4A424F4C
#define BL_LOGG_SIGNATURE 0x47474F4C

#define BL_APPID_BLFLOAD 22

#define BL_Major 0
#define BL_Minor 1
#define BL_AppBuild 1


#define BL_CHUNK 0x20000
#define BL_ZLIB_LEVEL 9


#ifdef _MSC_VER
#define packed
#else
#define packed __attribute__ ((__packed__))
typedef struct packed SYSTEMTIME_t {
    uint16_t wYear;
    uint16_t wMonth;
    uint16_t wDayOfWeek;
    uint16_t wDay;
    uint16_t wHour;
    uint16_t wMinute;
    uint16_t wSecond;
    uint16_t wMilliseconds;
} SYSTEMTIME;
#endif

typedef struct packed LOGG{
  uint32_t mSignature;
  uint32_t mHeaderSize;
  uint32_t mCRC;
  uint8_t  mApplicationID;
  uint8_t  mdwCompression;
  uint8_t  mApplicationMajor;
  uint8_t  mApplicationMinor;
  uint64_t mFileSize;
  uint64_t mUncompressedFileSize;
  uint32_t mObjectCount;
  uint8_t  mApplicationBuild;
  uint8_t  mReserved1;
  uint8_t  mReserved2;
  uint8_t  mReserved3;
  SYSTEMTIME mMeasurementStartTime;
  SYSTEMTIME mLastObjectTime;
  uint8_t  mReserved4[72];
} LOGG_t;

typedef struct packed VBLFileStatisticsEx_t {
  uint32_t      mStatisticsSize;               /* sizeof (VBLFileStatisticsEx) */
  uint8_t       mApplicationID;                /* application ID */
  uint8_t       mApplicationMajor;             /* application major number */
  uint8_t       mApplicationMinor;             /* application minor number */
  uint8_t       mApplicationBuild;             /* application build number */
  uint64_t      mFileSize;                     /* file size in bytes */
  uint64_t      mUncompressedFileSize;         /* uncompressed file size in bytes */
  uint32_t      mObjectCount;                  /* number of objects */
  uint32_t      mObjectsRead;                  /* number of objects read */
  SYSTEMTIME    mMeasurementStartTime;         /* measurement start time */
  SYSTEMTIME    mLastObjectTime;               /* last object time */
  uint32_t      mReserved[18];                 /* reserved */
} VBLFileStatisticsEx;

typedef struct packed VBLObjectHeaderBase_t {
  uint32_t  mSignature;
  uint16_t  mHeaderSize;
  uint16_t  mHeaderVersion;
  uint32_t  mObjectSize;
  uint32_t  mObjectType;
} VBLObjectHeaderBase;

typedef struct packed VBLObjectHeaderContainer_t {
  VBLObjectHeaderBase mBase;
  uint32_t            mCompressedflag;
  uint32_t            mReserved1;
  uint32_t            mDeflatebuffersize;
  uint32_t            mReserved2;
} VBLObjectHeaderContainer;

typedef struct packed VBLObjectHeader_t {
  VBLObjectHeaderBase mBase;
  uint32_t  mObjectFlags;
  uint16_t  mReserved;
  uint16_t  mObjectVersion;
  uint64_t  mObjectTimeStamp;
} VBLObjectHeader;

typedef struct packed VBLCANMessage_t {
  VBLObjectHeader mHeader;
  uint16_t        mChannel;
  uint8_t         mFlags;
  uint8_t         mDLC;
  uint32_t        mID;
  uint8_t         mData[8];
} VBLCANMessage;

typedef struct packed VBLCANMessage2_t {
  VBLObjectHeader mHeader;
  uint16_t        mChannel;
  uint8_t         mFlags;
  uint8_t         mDLC;
  uint32_t        mID;
  uint8_t         mData[8];
  uint32_t        mFrameLength;
  uint8_t         mBitCount;
  uint8_t         mReserved1;
  uint16_t        mReserved2;
} VBLCANMessage2;

typedef struct packed VBLCANErrorFrameExt_t {
    VBLObjectHeader    mHeader;                     /* object header */
    uint16_t           mChannel;                    /* application channel */
    uint16_t           mLength;                     /* CAN error frame length */
    uint32_t           mFlags;                      /* extended CAN error frame flags */
    uint8_t            mECC;                        /* error control code */
    uint8_t            mPosition;                   /* error position */
    uint8_t            mDLC;                        /* lower 4 bits: DLC from CAN-Core. Upper 4 bits: reserved */
    uint8_t            mReserved1;
    uint32_t           mFrameLengthInNS;            /* frame length in ns */
    uint32_t           mID;                         /* frame ID from CAN-Core */
    uint16_t           mFlagsExt;                   /* extended error flags */
    uint16_t           mReserved2;
    uint8_t            mData[8];                    /* Payload, only for CAN-Core */
} VBLCANErrorFrameExt;