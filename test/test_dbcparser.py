# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 00:07:35 2020

@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blfpy.dbcparser import dbc2code

if __name__ == "__main__":
    dbc = dbc2code(fn="../test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc")
    dbc.get_parser()

