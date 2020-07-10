# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:23:44 2020

@author: SNC6SI
"""

import numpy as np
from dbcparser import dbc2code
import blfpy


class blfread():
    
    def __init__(self, dbc=None, blf=None):
        self.__dbc = None
        self.__blf = None
        if dbc is not None:
            self.__dbc = dbc
        if blf is not None:
            self.__blf = blf
            
    @property
    def dbc(self):
        if self.__dbc is not None:
            return self.__dbc
        else:
            return ''

    @dbc.setter
    def dbc(self, dbc=None):
        if dbc is not None:
            self.__dbc = dbc
            
    @property
    def blf(self):
        if self.__blf is not None:
            return self.__blf
        else:
            return ''

    @blf.setter
    def blf(self, blf=None):
        if blf is not None:
            self.__blf = blf
    
    
    def collect_parser(self):
        self.parser = dbc2code(fn=self.__dbc)
        self.parser.do_parse()
    
    def unpack_data(self):
        # info = blfpy.readFileInfo()
        # print(info)
        d = blfpy.readFileData(self.__blf.encode('GBK'))
        if len(d[0])>0:
            self.d = d

    
    def parse_data(self):
        pass
    
    def save_data(self):
        pass
    
    def run_task(self):
        if self.__dbc is not None:
            self.collect_parser()
        if self.__blf is not None:
            self.unpack_data()
        if (self.__dbc is not None) and (self.__blf is not None):
            self.parse_data()
    
        

if __name__ == "__main__":
    a = blfread()
    a.dbc='test/dbc/ME7_PTCAN_CMatrix_190815_PVS.dbc'
    a.blf='20200608_IC321_500_快充测试009.blf'
    a.run_task()