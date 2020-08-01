# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:23:44 2020

@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>
"""

import os
import re
import numpy as np
from .dbcparser import dbc2code
from .blfc import read_data
# import matlab.engine


class blfload():


    def __init__(self, dbc=None, blf=None, signals=None):
        self.__dbc = None
        self.__blf = None
        if dbc is not None:
            self.__dbc = dbc
        if blf is not None:
            self.__blf = blf
        if signals is not None:
            self.signals = signals


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
    def read(self):
        if self.__dbc is not None:
            self.collect_parser()
        if self.__blf is not None:
            self.unpack_data()
        if (self.__dbc is not None) and (self.__blf is not None):
            # channel => canid => index
            if hasattr(self, 'signals'):
                if self.signals is not None:
                    self.get_data_index_specified()
                else:
                    self.get_data_index_default()
            else:
                self.get_data_index_default()
            self.detect_channel()
            self.parse_data()
            # return
            return self.can


    def collect_parser(self):
        self.parser = dbc2code(fn=self.__dbc)
        self.parser.get_parser()


    def unpack_data(self):
        # info = blfc.readFileInfo(self.__blf.encode('GBK'))
        # print(info)
        d = read_data(self.__blf.encode('GBK'))
        if len(d[0])>0:
            self.raw_data = d


    def get_data_index_default(self):
        # 0: 8-bytes
        # 1: id
        # 2: channel
        # 3: time
        # TODO: multiple blf data file
        '''
        self.data_index = {}
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
            self.data_index[ch] = ch_dict
        '''
        self.data_index = {}
        channels = np.unique(self.raw_data[2])
        for ch in channels:
            ch_dict = {}
            ch_idx = np.squeeze(np.argwhere(self.raw_data[2]==ch))
            can_ids = np.unique(self.raw_data[1][ch_idx])
            for can_id in can_ids:
                can_id_idx = ch_idx[np.argwhere(self.raw_data[1][ch_idx]==can_id)]
                ch_dict[can_id] = np.squeeze(can_id_idx)
            self.data_index[ch] = ch_dict


    def get_data_index_specified(self):
        # use closure maybe only for fun
        msg = self.parser.message
        def find_message_name(msg_rc):
            # tulpe (can_id, msg_name) shall be returned
            if isinstance(msg_rc, int):
                if msg_rc in msg.keys():
                    return msg[msg_rc]['canid'], msg[msg_rc]['name']
                else:
                    return None, None
            elif isinstance(msg_rc, str):
                name_tmp = [x[0] for x in msg.items() if msg_rc==x[1]['name']]
                if len(name_tmp)>0:
                    return name_tmp[0], msg[name_tmp[0]]['name'] # name_tmp is a list
                else:
                    pattern = '.*?(0x)?([0-9A-Fa-f]+).*'
                    id_hex = re.findall(pattern, msg_rc)[0][1]
                    return find_message_name(int(id_hex, 16))
            else:
                raise TypeError('Type: "%0" is not supported.' \
                                .format(type(msg_rc)))
        # print(find_message_name(abc))
        self.data_index = {}
        channels = np.unique(self.raw_data[2])
        signals = {}
        for ch in channels:
            ch_dict = {}
            ch_idx = np.squeeze(np.argwhere(self.raw_data[2]==ch))
            for key in self.signals.keys():
                can_id, msg_name = find_message_name(key)
                if can_id is not None:
                    signals[can_id] = self.signals[key]
                    can_id_idx = ch_idx[np.argwhere(self.raw_data[1][ch_idx]==can_id)]
                    if can_id_idx.size > 0:
                        ch_dict[can_id] = np.squeeze(can_id_idx)
            self.data_index[ch] = ch_dict
            self.signals_checked = signals


    def detect_channel(self):
        # id from dbc
        dbc_id = np.array(list(self.parser.message.keys()))
        # channel and id in data
        self.intersection = {}
        for ch in self.data_index.keys():
            ids = np.intersect1d(dbc_id, list(self.data_index[ch].keys()))
            self.intersection[ch] = ids
        '''
        ch_max = max(self.intersection.keys(),
                     key=(lambda x: self.intersection[x].size))
        '''


    def parse_data(self):
        ch_max = max(self.intersection.keys(),
                     key=(lambda x: self.intersection[x].size))
        data_index = self.data_index[ch_max]
        ids = self.intersection[ch_max]
        self.parsed_data = {}
        for msg_id in ids:
            msg_p = {}
            idx = data_index[msg_id]
            msg_p['ctime'] = self.raw_data[3][idx]
            bb = self.raw_data[0][idx].astype(np.uint)
            message = self.parser.message[msg_id]
            # k: signal name,  v: signal info dict
            for k, v in message['signal'].items():
                if hasattr(self, 'signals_checked'):
                    if k in self.signals_checked[msg_id]:
                        msg_p[k] = eval(v['pycode'])
                else:
                    msg_p[k] = eval(v['pycode'])
            self.parsed_data[message['name']] = msg_p


    def save_data(self, file_name=None, file_format='mat'):
        # TODO: rewrite as a interface, will also save as mdf
        # default file_name
        if file_name is None:
            p = os.path.abspath(self.blf)
            file_name = ''.join((os.path.splitext(p)[0], '.mat'))
        if file_format=='mat':
            mdict = {'can': self.parsed_data}
            from scipy.io import savemat
            savemat(file_name,
                    mdict,
                    long_field_names=True,
                    do_compression=True)
        elif file_format=='mdf':
            pass
        else:
            raise ValueError(f"\"{file_format}\" is not supported.")


    '''
    def __getattr__(self, attr):
        # if not hasattr(self, 'eng'):
        #     self.__matlab__()
        if attr=='eng':
            self.__matlab__()
            return self.eng
        else:
            if hasattr(self.eng, attr):
                return getattr(self.eng, attr)
            else:
                raise ValueError('There is no property: "%s".'%attr)


    def __matlab__(self):
        # hasattr will cause problem, if matlab was shut down
        eng_rc = matlab.engine.find_matlab()
        if len(eng_rc)==0:
            eng = matlab.engine.start_matlab(option='-nodesktop')
        else:
            eng = matlab.engine.connect_matlab(eng_rc[0])
        self.eng = eng
    '''
    

if __name__ == "__main__":
    bl = blfload(dbc='test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc',
                 blf='20200608_IC321_500_快充测试009.blf')
    bl.signals = {'VBU_BMS_0x100': ['VBU_BMS_PackU',
                                    'VBU_BMS_PackI',
                                    'VBU_BMS_State'],
                  'VBU_BMS_0x102': ['VBU_BMS_RealSOC',
                                    'VBU_BMS_PackDispSoc'],
                  'VBU_BMS_0x513': ['VBU_BMS_MaxTemp',
                                    'VBU_BMS_MinTemp']}
    # del bl.signals
    # channel = None
    bl.read()
    # bl.plot(matlab.double(bl.can['VBU_BMS_0x100']['ctime'].tolist()),
    #         matlab.double( bl.can['VBU_BMS_0x100']['VBU_BMS_PackU'].tolist()))
    # bl.grid('on', nargout=0)
    # bl.eng.exit()