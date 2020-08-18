# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:06:39 2020

@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>
"""

import re
import math
import numpy as np

class dbc2code():

    __BO_Blk_RE = re.compile(r'BO_ \d+ [a-zA-Z].+?\n\n', re.DOTALL)
    __BO_RE = re.compile(r"BO_ (?P<canid>\d+) (?P<name>\w+)")
    __SG_RE = re.compile(r"\s*SG_ " + \
                         r"(?P<name>\w+) : " + \
                         r"(?P<start>\d+)\|" + \
                         r"(?P<length>\d+)@" + \
                         r"(?P<endian>[01])\+? \(" + \
                         r"(?P<gain>\d+(\.\d*)?)," + \
                         r"(?P<offset>-?\d+(\.\d*)?)\) \[" + \
                         r"(?P<phymin>-?\d+(\.\d*)?(E\+\d+)?)\|" + \
                         r"(?P<phymax>-?\d+(\.\d*)?(E\+\d+)?)\] \"" + \
                         r"(?P<unit>.*?)\"\s+" + \
                         r"(?P<reciever>.*)")
    __VAL_RE = re.compile(r'VAL_ (\d+) (\w+) ((?:\d+ ".+?")+?) ;')
    __VAL_INTERN_RE = re.compile(r'(\d+) "(.*?)"')
    __BA_GenMsgCycleTime_RE = re.compile(r"BA_ \"GenMsgCycleTime\" BO_ " + \
                                         r"(\d+) (\d+);")
    __BITMATRIX = np.flip(np.arange(64).reshape(8, 8), 1).reshape(64,)
    __BB = 'bb'


    def __init__(self, fn=None):
        if fn is None:
            self.__dbc_raw = None
        else:
            with open(fn, 'rt', encoding='GBK') as f:
                self.__dbc_raw = f.read()


    def get_parser(self):
        self.__BO_blks = self.__BO_Blk_RE.findall(self.__dbc_raw)
        self._get_enum()
        self._get_GenMsgCycleTime()
        self.message = {}
        for BO in self.__BO_blks:
            lines = BO.split('\n')
            BO_dict = self.__BO_RE.match(lines[0]).groupdict()
            BO_dict['canid'] = int(BO_dict['canid'])
            BO_dict['canid_hex'] = format(BO_dict['canid'], 'X')
            if BO_dict['canid'] in self.periods.keys():
                BO_dict['period'] = self.periods[BO_dict['canid']]
            else:
                BO_dict['period'] = None
            SG_dicts = {}
            for SG in lines[1:]:
                if len(SG) > 0:
                    SG_dict = self.__SG_RE.match(SG).groupdict()
                    SG_dict['start'] = int(SG_dict['start'])
                    SG_dict['length'] = int(SG_dict['length'])
                    SG_dict['endian'] = int(SG_dict['endian'])
                    SG_dict['gain'] = float(SG_dict['gain'])
                    SG_dict['offset'] = float(SG_dict['offset'])
                    SG_dict['phymin'] = float(SG_dict['phymin'])
                    SG_dict['phymax'] = float(SG_dict['phymax'])
                    SG_dict['reciever'] = SG_dict['reciever'].split(',')
                    # involke parse method
                    SG_dict['sigmat'] = \
                        self._parser_internal_info2matrix(SG_dict)
                    SG_dict['pycode_raw2si'], \
                    SG_dict['pycode_si2phy'] = \
                        self._parser_internal_matrix2py(SG_dict)
                    SG_dict['mat_value2bytes'], \
                    SG_dict['mat_raw2bytes'] = \
                        self._parser_internal_pack(SG_dict)
                    # enum
                    if BO_dict['canid'] in self.enums.keys():
                        if SG_dict['name'] in \
                            self.enums[BO_dict['canid']].keys():
                            SG_dict['enum'] = \
                                self.enums[BO_dict['canid']][SG_dict['name']]
                    # 
                    SG_dicts[SG_dict['name']] = SG_dict
            BO_dict['signal'] = SG_dicts
            # map
            mapping_v2s = {}
            mapping_s2v = {}
            for i, name in enumerate(BO_dict['signal'].keys()):
                v = f'v{i}'
                mapping_v2s[v] = name
                mapping_s2v[name] = v
            BO_dict['mapping_v2s'] = mapping_v2s
            BO_dict['mapping_s2v'] = mapping_s2v
            # merge
            mat_raw2pack = [[]] * 8
            for k, v in BO_dict['signal'].items():
                for i, s in enumerate(v['mat_raw2bytes']):
                    if len(s):
                        # find rep str
                        rep = BO_dict['mapping_s2v'][v['name']]
                        # do rep
                        ss = re.sub('rr', rep, s)
                        # merge
                        if len(mat_raw2pack[i]):
                            ss = f" | {ss}"
                            mat_raw2pack[i] += ss
                        else:
                            mat_raw2pack[i] = ss
            BO_dict['mat_raw2pack'] = mat_raw2pack
            self.message[BO_dict['canid']] = BO_dict


    def _parser_internal_info2matrix(self, info):
        if not info['endian']:
            # motorola
            bitend_idx = np.argwhere(self.__BITMATRIX==info['start'])
            bitstart_idx = bitend_idx + info['length'] - 1
        else:
            # intel
            bitstart_idx = np.argwhere(self.__BITMATRIX==info['start'])
            bitend_idx = \
                np.argwhere(self.__BITMATRIX==(info['start']+info['length']-1))
        
        bitend_bytepos = math.floor(bitend_idx/8)
        bitstart_bytepos = math.floor(bitstart_idx/8)
        
        bit_temp = (bitend_idx + 1)%8
        if bit_temp:
            bitend_bitpos = 8 - bit_temp
        else:
            bitend_bitpos = 0
            
        bit_temp = (bitstart_idx + 1)%8
        if bit_temp:
            bitstart_bitpos = 8 - bit_temp
        else:
            bitstart_bitpos = 0
            
        loopnum = abs(bitstart_bytepos - bitend_bytepos) + 1
        sigmat = np.zeros((loopnum, 5), dtype=int)
        # which byte
        # start bit pos this line
        # end bit pos this line
        # bit cnt this line
        # bit cnt sum previous line
        if loopnum == 1:
            sigmat[0, 0] = bitstart_bytepos
            sigmat[0, 1] = bitstart_bitpos
            sigmat[0, 2] = bitend_bitpos
            sigmat[0, 3] = sigmat[0, 2] - sigmat[0, 1] + 1
            sigmat[0, 4] = 0
        else:
            for i in range(loopnum):
                if i == 0:
                    sigmat[i, 0] = bitstart_bytepos
                    sigmat[i, 1] = bitstart_bitpos
                    sigmat[i, 2] = 7
                    sigmat[i, 3] = sigmat[i, 2] - sigmat[i, 1] + 1
                    sigmat[i, 4] = 0
                elif i < loopnum-1:
                    if not info['endian']:
                        sigmat[i, 0] = sigmat[i-1, 0] - 1
                    else:
                        sigmat[i, 0] = sigmat[i-1, 0] + 1
                    sigmat[i, 1] = 0
                    sigmat[i, 2] = 7
                    sigmat[i, 3] = sigmat[i, 2] - sigmat[i, 1] + 1
                    sigmat[i, 4] = sigmat[i-1, 4] + sigmat[i-1, 3]
                else:
                    sigmat[i, 0] = bitend_bytepos
                    sigmat[i, 1] = 0
                    sigmat[i, 2] = bitend_bitpos
                    sigmat[i, 3] = sigmat[i, 2] - sigmat[i, 1] + 1
                    sigmat[i, 4] = sigmat[i-1, 4] + sigmat[i-1, 3]
        return sigmat


    def _parser_internal_matrix2py(self, info):
        sigmat = info['sigmat']
        gain = info['gain']
        offset = info['offset']
        # unpack
        loopnum = sigmat.shape[0]
        si = ''
        for j in range(loopnum):
            if j > 0:
                si = ' + ' + si
            s = self.__BB + '[:,' + str(sigmat[j, 0]) + ']'
            s =  '(' + s + '>>' + str(sigmat[j, 1]) + ')'
            s = '(' + s + '&(' + str(2**sigmat[j, 3]-1) + '))'
            if sigmat[j, 4]:
                s = '(' + str(2**sigmat[j, 4]) + ')*' + s
            si = s + si
        si = '(' + si + ')'
        # gain offset
        if gain!=1:
            phy = 'si *' + str(gain)
        else:
            phy = 'si'
        if offset:
            phy += '+' + str(offset)
        return si, phy


    def _parser_internal_pack(self, info):
        # mat_value2bytes = [''] * 8
        mat_value2bytes = 0
        mat_raw2bytes = [''] * 8
        rr = "rr"
        mat = info['sigmat']
        loopnum = mat.shape[0]
        for i in range(loopnum):
            mat_raw2bytes[mat[i,0]] = \
                f"((rr>>{mat[i,4]})&{2**mat[i,3]-1})<<{mat[i,1]}"
        return mat_value2bytes, mat_raw2bytes


    def _get_enum(self):
        val_raw = self.__VAL_RE.findall(self.__dbc_raw)
        self.enums = {}
        for val in val_raw:
            canid = int(val[0])
            signal = val[1]
            enum = dict(self.__VAL_INTERN_RE.findall(val[2]))
            if canid not in self.enums.keys():
                self.enums[canid] = {}
            self.enums[canid][signal] = enum


    def _get_GenMsgCycleTime(self):
        cycle_time_canid_str = \
            self.__BA_GenMsgCycleTime_RE.findall(self.__dbc_raw)
        cycle_times_canid_dec = list(map(lambda x: [int(x[0]), int(x[1])],
                                         cycle_time_canid_str))
        self.periods = dict(cycle_times_canid_dec)


    def to_dict(self, level='signal'):
        if not hasattr(self, 'message'):
            return None
        else:
            import copy
            message = copy.deepcopy(self.message)
            if level == 'message':
                for k in message.keys():
                    message[k] = ''
                return message
            elif level == 'signal':
                for k, v in message.items():
                    message[k] = list(v['signal'].keys())
                return message
            elif level == 'all':
                for k, v in message.items():
                    for kk, vv in v['signal'].items():
                        del vv['sigmat']
                        del vv['pycode_raw2si']
                        del vv['pycode_si2phy']
                        if 'enum' in vv.keys():
                            del vv['enum']
                return message
            else:
                return None


    def get_name_canid_mapping(self):
        mapping = {}
        for k, v in self.message.items():
            mapping[v['name']] = v['canid']
        return mapping


if __name__ == "__main__":
    dbc = dbc2code(fn="test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc")
    dbc.get_parser()
    message_name = dbc.get_message_name()