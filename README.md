# blfpy
> A tool for better automating CAN-data proccessing in python

> Cross-platform: both windows & linux

- read communication protocol in dbc-format
- read data from blf and mdf(only version 3.00 is tested)
- convert data to structured data
- save structured data as matlab(mat) format

**additonal**
- convert blf to mdf for fun
- convert mdf to blf for sharing data to colleagues who are not familiar with mdf viewer

## setup
- You will have a distributed version with its name like: 
blfpy-x.x.x-cp37-cp37m-win_amd64.whl (this one is for python 3.7)
- Make the folder where whl file resides in the current folder
- Both in pip or conda environment, type:

```shell
pip install blfpy-x.x.x-cp37-cp37m-win_amd64.whl
```

- blfpy has these third party dependencies: numpy, scipy
- blfpy for following python version is provided: 3.7, 3.8


## usage
> Although you can use blfpy in pip environment, it is strongly recommended,
that you use it in conda environment.

**blf as original**

- import (better import numpy simultanously as well)
```python
>>> import numpy as np
>>> from blfpy.blfload import blfload
```

- create blfload object with arguments (dbc, blf)
```python
>>> bl = blfload(dbc='PTCAN_CMatrix_V1.7_PT_VBU.dbc',
                 blf='20200608_IC321_500_009.blf')
```
- or create blf object without arguments, and fill them afterwards
```python
>>> bl = blfload()
>>> bl.dbc = 'PTCAN_CMatrix_V1.7_PT_VBU.dbc'
>>> bl.blf = '20200608_IC321_500_009.blf'
```
- without signal selection, all the message in dbc will be parsed and returned
- signals can be selected as follows in form of dict: 
(signal selection can make parsing phase much faster)

```python
>>> bl.signals = {'VBU_BMS_0x100': ['VBU_BMS_PackU',
                                    'VBU_BMS_PackI',
                                    'VBU_BMS_State'],
                  'VBU_BMS_0x102': ['VBU_BMS_RealSOC',
                                    'VBU_BMS_PackDispSoc'],
                  'VBU_BMS_0x513': ['VBU_BMS_MaxTemp',
                                    'VBU_BMS_MinTemp']}
```

- call blfload.read method will invoke following functionalities:
    - READ infos from dbc
    - EXTRACT raw data from blf
    - DETERMINE which data will be parsed (default: all) and their indices in raw data
    - DETECT channel: automatically make channel mapping from dbc-infos to data (will also be a user interface later)
    - PARSE data and create a dict in python workspace

```python
>>> bl.read()
```

- sava_data to mat-file or mdf-file
- if filename is omitted, the saved file will have the save basename of the original file
```python
>>> bl.save_data(filename='None'/'xxx', file_format='mat'/'mdf')
```

**mdf as original**

- similar to blf, the procedure is as follows
- in order to convert loaded mdf to blf, a communication protocol(dbc-file) must be specified
```python
>>> import numpy as np
>>> from blfpy.mdfload import mdfread
>>> mdf_file = r'../test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
>>> m = mdfread(mdf=mdf_file)
>>> m.read()
>>> m.save_data(file_format='mat')
>>> m.save_data(file_format='blf',
                dbc='../test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc')
```


## TODO
- maybe or maybe not plot in Matlab

