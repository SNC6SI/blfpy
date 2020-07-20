# blfpy
> A tool for better automating CAN-data proccessing in python
- read communication protocol in dbc-format
- read can messges in blf
- convert them to structured data
- save structured data as matlab/mdf format

## setup
- You will have a distributed version with its name like: 
blfpy-0.2.0-cp37-cp37m-win_amd64.whl (this one is for python 3.7)
- Make the folder where whl file resides in the current folder
- Both in pip or conda environment, type:

```shell
pip install blfpy-0.2.0-cp37-cp37m-win_amd64.whl
```

## usage
> Although you can use blfpy in pip environment, it is strongly recommended,
that you use it in conda environment.

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
- or create blf object without argument, and fill them afterwards
```python
>>> bl = blfload()
>>> bl.dbc = 'PTCAN_CMatrix_V1.7_PT_VBU.dbc'
>>> bl.blf = '20200608_IC321_500_009.blf'
```
- without signal selection, all the message in dbc will be parsed and returned
- signals can be selected as follows in form of dict: 
(signal selection can make parse much faster)

```python
>>> bl.signals = {'VBU_BMS_0x100': ['VBU_BMS_PackU',
                                    'VBU_BMS_PackI',
                                    'VBU_BMS_State'],
                  'VBU_BMS_0x102': ['VBU_BMS_RealSOC',
                                    'VBU_BMS_PackDispSoc'],
                  'VBU_BMS_0x513': ['VBU_BMS_MaxTemp',
                                    'VBU_BMS_MinTemp']}
```
- sava_data to mat-file (data format of matlab)
- this method will be changed later for also save as mdf
```python
>>> bl.save_data()
```

## TODO
- save as mdf format
- maybe or maybe not plot in Matlab
- ~~maybe or maybe not relevant scripts to use the parsed data~~
