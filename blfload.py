# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:23:44 2020

@author: SNC6SI
"""

import numpy as np
from dbcparser import dbc2code


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
    
    
    def collect_parser(self):
        pass
    
    def unpack_data(self):
        pass
    
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