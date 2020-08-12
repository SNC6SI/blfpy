# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:16:44 2020

@author: SNC6SI: Shen,Chenghao <snc6si@gmail.com>
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy
from blfpy.blfload import blfload

if __name__ == "__main__":
    bl = blfload(dbc=os.path.join(os.path.dirname(__file__), 
                                  'dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc'),
                 blf=os.path.join(os.path.dirname(__file__),
                                  '20200608_IC321_500_快充测试009.blf'))
    bl.signals = {'VBU_BMS_0x100': ['VBU_BMS_PackU',
                                    'VBU_BMS_PackI',
                                    'VBU_BMS_State'],
                  'VBU_BMS_0x102': ['VBU_BMS_RealSOC',
                                    'VBU_BMS_PackDispSoc'],
                  'VBU_BMS_0x513': ['VBU_BMS_MaxTemp',
                                    'VBU_BMS_MinTemp']}
    del bl.signals
    can = bl.read()
    # bl.plot(matlab.double(bl.can['VBU_BMS_0x100']['ctime'].tolist()),
    #         matlab.double( bl.can['VBU_BMS_0x100']['VBU_BMS_PackU'].tolist()))
    # bl.grid('on', nargout=0)
    # bl.eng.exit()


    bl.save_data(file_format='mdf')
    # bl.w.data.tofile('abc16.dat')