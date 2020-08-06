# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:33:00 2020

@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy
from blfpy.mdfload import mdfread



if __name__ == "__main__":
    # mdf_file = r'abc6.dat'
    mdf_file = r'test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
    m = mdfread(mdf=mdf_file)
    m.read() 