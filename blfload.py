# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:23:44 2020

@author: SNC6SI
"""

import os
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


    # =========================================================================
    # property
    # =========================================================================
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

    # TODO: 
        # blf can be str or list of str
        # with same dbc but different raw_data, load dbcparser only once
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


    @property
    def can(self):
        if hasattr(self, 'parsed_data'):
            return self.parsed_data
        else:
            return None


    @can.setter
    def can(self, value=None):
        pass

    # =========================================================================
    # methods
    # =========================================================================

    def collect_parser(self):
        self.parser = dbc2code(fn=self.__dbc)
        self.parser.do_parse()


    def unpack_data(self):
        # info = blfpy.readFileInfo(self.__blf.encode('GBK'))
        # print(info)
        d = blfpy.readFileData(self.__blf.encode('GBK'))
        if len(d[0])>0:
            self.raw_data = d


    def get_data_info(self):
        # 0: 8-bytes
        # 1: id
        # 2: channel
        # 3: time
        # TODO: multiple blf data file
        '''
        self.data_info = {}
        channels = np.unique(self.raw_data[2])
        for ch in channels:
            ch_dict = {}
            ch_idx = np.argwhere(self.raw_data[2]==ch)
            can_ids = np.unique(self.raw_data[1][ch_idx])
            for can_id in can_ids:
                can_id_idx = \
                    np.argwhere(np.logical_and(self.raw_data[1]==can_id,
                                                self.raw_data[2]==ch))
                ch_dict[can_id] = np.squeeze(can_id_idx)
            self.data_info[ch] = ch_dict
        '''
        self.data_info = {}
        channels = np.unique(self.raw_data[2])
        for ch in channels:
            ch_dict = {}
            ch_idx = np.squeeze(np.argwhere(self.raw_data[2]==ch))
            can_ids = np.unique(self.raw_data[1][ch_idx])
            for can_id in can_ids:
                can_id_idx = ch_idx[np.argwhere(self.raw_data[1][ch_idx]==can_id)]
                ch_dict[can_id] = np.squeeze(can_id_idx)
            self.data_info[ch] = ch_dict


    def parse_all(self):
        ch_max = max(self.intersection.keys(),
                     key=(lambda x: self.intersection[x].size))
        data_info = self.data_info[ch_max]
        ids = self.intersection[ch_max]
        self.parsed_data = {}
        for msg_id in ids:
            msg_p = {}
            
            idx = data_info[msg_id]
            msg_p['ctime'] = self.raw_data[3][idx]
            bb = self.raw_data[0][idx].astype(np.uint)
            message = self.parser.message[msg_id]
            # k: signal name,  v: signal info dict
            for k, v in message['signal'].items():
                msg_p[k] = eval(v['pycode'])
            self.parsed_data[message['name']] = msg_p
            
    
    
    def detect_channel(self):
        # id from dbc
        dbc_id = np.array(list(self.parser.message.keys()))
        # channel and id in data
        self.intersection = {}
        for ch in self.data_info.keys():
            ids = np.intersect1d(dbc_id, list(self.data_info[ch].keys()))
            self.intersection[ch] = ids
        '''
        ch_max = max(self.intersection.keys(),
                     key=(lambda x: self.intersection[x].size))
        '''


    def save_data(self, mat_fn=None):
        from scipy.io import savemat
        if mat_fn is None:
            p = os.path.abspath(self.blf)
            mat_fn = ''.join((os.path.splitext(p)[0], '.mat'))
            mdict = {'can': self.parsed_data}
        savemat(mat_fn, mdict, long_field_names=True, do_compression=True)


    def run_task(self):
        if self.__dbc is not None:
            self.collect_parser()
        if self.__blf is not None:
            self.unpack_data()
        if (self.__dbc is not None) and (self.__blf is not None):
            self.get_data_info()
            self.detect_channel()
            self.parse_all()
    


if __name__ == "__main__":
    bl = blfread(dbc='test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc',
                 blf='20200608_IC321_500_快充测试009.blf')
    bl.run_task()