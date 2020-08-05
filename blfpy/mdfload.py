# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:49:28 2020

@author: SNC6SI: Shen,Chenghao <snc6si@gmail.com>
"""

from struct import unpack, pack, calcsize
import warnings
import math
import re
import numpy as np
from scipy import interpolate


class mdfread:

    __SIGNAME_RE = re.compile(r'(\w+)\\?.*')

    def __init__(self, mdf=None):
        if mdf is not None:
            self.mdf = mdf


    def read(self):
        self.data = np.fromfile(self.mdf, dtype=np.uint8)
        
        self.idblock = self.IDBLOCK(self.data)
        if self.idblock.endian:
            self.endian = '>'
        else:
            self.endian = '<'
        self.pointer = {'dg':[],
                        'cg':[],
                        'cn':[],
                        'cc':[],
                        'dt':[]}
        self.hdblock = self.HDBLOCK(self.data, self.endian)


        # DG
        if self.hdblock.num_dg_blocks:
            dgblock = self.DGBLOCK(self.data, self.endian, self.hdblock.p_dg_block)
            self.dgblocks = [dgblock]
            self.pointer['dg'] += [self.hdblock.p_dg_block]
            while dgblock.p_dg_block:
                self.pointer['dg'] += [dgblock.p_dg_block]
                dgblock = self.DGBLOCK(self.data, self.endian, dgblock.p_dg_block)
                self.dgblocks += [dgblock]


        # CG
        for dgblock in self.dgblocks:
            if dgblock.num_cg_blocks:
                cgblock = self.CGBLOCK(self.data, self.endian, dgblock.p_cg_block)
                cgblocks = [cgblock]
                self.pointer['cg'] += [dgblock.p_cg_block]
                while cgblock.p_cg_block:
                    self.pointer['cg'] += [dgblock.p_cg_block]
                    cgblock = self.CGBLOCK(self.data, self.endian, cgblock.p_cg_block)
                    cgblocks += [cgblock]
                dgblock.cgblocks = cgblocks


        # CN
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                cnblock = self.CNBLOCK(self.data, self.endian, cgblock.p_cn_block)
                cnblocks = [cnblock]
                self.pointer['cn'] += [cgblock.p_cn_block]
                while cnblock.p_cn_block:
                    self.pointer['cn'] += [cnblock.p_cn_block]
                    cnblock = self.CNBLOCK(self.data, self.endian, cnblock.p_cn_block)
                    cnblocks += [cnblock]
                cgblock.cnblocks = cnblocks


        # DT
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                if dgblock.num_record_ids==0:
                    dgblock.records = self.DT(self.data,
                                              dgblock.num_record_ids,
                                              dgblock.p_records,
                                              cgblock.record_size,
                                              cgblock.num_records)
                    self.pointer['dt'] += [dgblock.p_records]


        # CC
        for dgblock in self.dgblocks:
             for cgblock in dgblock.cgblocks:
                 for cnblock in cgblock.cnblocks:
                     # print(cnblock.signal_name)
                     cnblock.ccblock = \
                         self.CCBLOCK(self.data, self.endian, cnblock.p_cc_block)
                     self.pointer['cc'] += [cnblock.p_cc_block]


        # raw
        for dgblock in self.dgblocks:
            '''
            pay attention:
            - bb, when merge these loops
            - record id is now not considered yet
            '''
            bb = dgblock.records.mat
            for cgblock in dgblock.cgblocks:
                for cnblock in cgblock.cnblocks:
                    byte_start = cnblock.byte_offset + \
                                 math.floor(cnblock.bit_start/8)
                    byte_end = cnblock.byte_offset + \
                        math.ceil((cnblock.bit_start + cnblock.bit_length)/8)
                    byte_length = byte_end-byte_start
                    
                    # dt: data type
                    # dl: data length
                    if byte_length<=8:
                        if cnblock.signal_data_type==0:
                            dt = 'u'
                            dl = str(2**(math.ceil(math.log2(byte_length))))
                        elif cnblock.signal_data_type==1:
                            dt = 'i'
                            dl = str(2**(math.ceil(math.log2(byte_length))))
                        elif cnblock.signal_data_type==2:
                            dt = 'f'
                            dl = '4'
                        elif cnblock.signal_data_type==3:
                            dt = 'f'
                            dl = '8'
                        else:
                            raise ValueError('data_type:%u for signal %s is not supported.'% \
                                             (cnblock.signal_data_type, cnblock.signal_name))
                        view = self.endian + 'u' + str(2**(math.ceil(math.log2(byte_length))))
                        raw = bb[:, byte_start:byte_end].copy().view(view)
                        raw = (raw>>(cnblock.bit_start%8))&np.uint64((2**cnblock.bit_length-1))

                        view = self.endian + dt + dl
                        raw = bb[:, byte_start:byte_end].copy().view(view)
                        
                    else:
                        raw = bb[:, byte_start:byte_end].copy()
                    cnblock.raw = raw


        # phy
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                for cnblock in cgblock.cnblocks:
                    raw = cnblock.raw
                    f_id = cnblock.ccblock.formula_id
                    param = cnblock.ccblock.parameters
                    value = eval(cnblock.ccblock.pycode)
                    cnblock.value = value


        self.__read_post_pointer()
        self.__read_post_parsed_data()


    def __read_post_pointer(self):
        self.pointer['dg'].sort()
        self.pointer['cg'].sort()
        self.pointer['cn'].sort()
        self.pointer['cc'].sort()
        self.pointer['dt'].sort()


    def __read_post_parsed_data(self):
        self.parsed_data = {}
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                #
                time = None
                # loop 1: find time
                for cnblock in cgblock.cnblocks:
                    # cn_type 0:data, 1:time
                    if cnblock.cn_type==0:
                        continue
                    else:
                        time = cnblock.value
                        break
                # if time is not found, skip this CG
                if time is None:
                    continue
                # loop 2: get data
                for cnblock in cgblock.cnblocks:
                    signal = {}
                    if cnblock.signal_name=='time':
                        continue
                    else:
                        signal_name = \
                            self.__SIGNAME_RE.findall(cnblock.signal_name)[0]
                    signal['raw'] = cnblock.raw
                    signal['value'] = cnblock.value
                    signal['time'] = time
                    self.parsed_data[signal_name] = signal


    class IDBLOCK:
        """
        useful infos:
            - version
        """
    
        def __init__(self, data):
            p = 0
            d = data
            
            self.byte_order = np.squeeze(d[p+24:p+26].view('u2'))
            if self.byte_order:
                E = '>'
            else:
                E = '<'
                
            fmt = E + '8s8s8sHHH'
            size = calcsize(fmt)
            
            file_identifier, \
            format_identifier, \
            program_identifier, \
            _, \
            floating_point_format, \
            version = unpack(fmt, d[p:p+size].tobytes())
                
            self.file_identifier = file_identifier.decode().rstrip('\x00')
            self.format_identifier = format_identifier.decode().rstrip('\x00')
            self.program_identifier = program_identifier.decode().rstrip('\x00')
            self.floating_point_format = floating_point_format
            self.version = version
            
            if self.version!=300:
                warnings.warn('Only MDF version 3.00 is supported.', UserWarning)
            
            '''
            following properties are not used in version 3.00
            comment them for now
            '''
            # self.code_page_number = np.squeeze(data[p+30:p+32].view(E+'u2'))
            # self.standard_flags = np.squeeze(data[p+60:p+62].view(E+'u2'))
            # self.custom_flags = np.squeeze(data[p+62:p+64].view(E+'u2'))


        @property
        def endian(self):
            return self.byte_order


    class HDBLOCK:
        """
        useful infos:
            - p_dg_block
            - num_dg_blocks
        """
        
        def __init__(self, data, E):
            p = 64
            d = data # +208 for 3.30 and is not implemented
            
            fmt = E + '2sHIIIH10s8s32s32s32s32s'
            size = calcsize(fmt)
    
            block_type, \
            block_size, \
            p_dg_block, \
            p_tx_block, \
            p_pr_block, \
            num_dg_blocks, \
            record_date, \
            record_time, \
            author_name, \
            org_dept_name, \
            project_name, \
            subject_name = unpack(fmt, d[p:p+size].tobytes())
            
            self.block_type = block_type.decode().rstrip('\x00')
            self.block_size = block_size
            self.p_dg_block = p_dg_block
            self.p_tx_block = p_tx_block
            self.p_pr_block = p_pr_block
            self.num_dg_blocks = num_dg_blocks
            self.record_date = record_date.decode() # DD:MM:YYYY
            self.record_time = record_time.decode() # HH:MM:SS
            self.author_name = author_name.decode().rstrip('\x00')
            self.org_dept_name = org_dept_name.decode().rstrip('\x00')
            self.project_name = project_name.decode().rstrip('\x00')
            self.subject_name = subject_name.decode().rstrip('\x00')


    class DGBLOCK:
        """
        useful infos:
            - p_dg_block
            - p_cg_block
            - num_cg_blocks
            ? num_record_ids
        """
        
        def __init__(self, data, E, p):
            d = data
            
            fmt = E + '2sHIIIIHH'
            size = calcsize(fmt)
            
            block_type, \
            block_size, \
            p_dg_block, \
            p_cg_block, \
            _, \
            p_records, \
            num_cg_blocks, \
            num_record_ids = unpack(fmt, d[p:p+size].tobytes())
    
            self.block_type = block_type.decode().rstrip('\x00')
            self.block_size = block_size
            self.p_dg_block = p_dg_block
            self.p_cg_block = p_cg_block
            self.p_records = p_records
            self.num_cg_blocks = num_cg_blocks
            self.num_record_ids = num_record_ids


    # DATARECORDS
    class DT:
        
        def __init__(self, data, record_id, p, s, n):
            # p: p_records
            # s: record_size
            # n: num_records
            d = data
            if record_id==0:
                self.mat = d[p:p+s*n].reshape((n,s))
        
            

    class CGBLOCK:
        """
        useful infos:
            - p_cg_block
            - p_cn_block
            - num_cn_blocks
            ? record_id
            ? num_record_ids
            - record_size
            - num_records
        """
        
        def __init__(self, data, E, p):
            d = data
            
            fmt = E + '2sHIIIHHHI'
            size = calcsize(fmt)
            
            block_type, \
            block_size, \
            p_cg_block, \
            p_cn_block, \
            p_tx_block, \
            record_id, \
            num_cn_blocks, \
            record_size, \
            num_records = unpack(fmt, d[p:p+size].tobytes())
            
            self.block_type = block_type.decode().rstrip('\x00')
            self.block_size = block_size
            self.p_cg_block = p_cg_block
            self.p_cn_block = p_cn_block
            self.p_tx_block = p_tx_block
            self.record_id = record_id
            self.num_cn_blocks = num_cn_blocks
            self.record_size = record_size
            self.num_records = num_records


    class CNBLOCK:
        """
        useful infos:
            - signal_name
            - signal_description
            - signal_data_type
            - sample_rate
            
            - p_cn_block
            - p_cc_block
            
            - cn_type
            
            - bit_start
            - bit_length
            - byte_offset
        """
    
        def __init__(self, data, E, p):
            d = data
            
            fmt = E + '2sHIIIIIH32s128sHHHHdddIIH'
            size = calcsize(fmt)
            
            block_type, \
            block_size, \
            p_cn_block, \
            p_cc_block, \
            _, \
            _, \
            p_tx_block, \
            cn_type, \
            signal_name, \
            signal_description, \
            bit_start, \
            bit_length, \
            signal_data_type, \
            bool_value_range, \
            min_value_range, \
            max_value_range, \
            sample_rate, \
            p_unique_name, \
            _, \
            byte_offset = unpack(fmt, d[p:p+size].tobytes())
            
            self.block_type = block_type.decode().rstrip('\x00')
            self.block_size = block_size
            self.p_cn_block = p_cn_block
            self.p_cc_block = p_cc_block
            self.p_tx_block = p_tx_block
            # cn_type: 0=data, 1=time
            self.cn_type = cn_type
            self.signal_name = signal_name.decode().rstrip('\x00')
            try:
                self.signal_description = signal_description.rstrip('\x00') \
                                          .rsplit(b'\r\n')[0].decode()
            except:
                self.signal_description = ''
            self.bit_start = bit_start
            self.bit_length = bit_length
            self.signal_data_type = signal_data_type
            self.bool_value_range = bool_value_range
            self.min_value_range = min_value_range
            self.max_value_range = max_value_range
            self.sample_rate = sample_rate
            self.p_unique_name = p_unique_name
            self.byte_offset = byte_offset


    class CCBLOCK:
        """
        useful infos:
            - phy_unit
            - formula_id
            - parameters
        """
        
        def __init__(self, data, E, p):
            d = data
    
            self.formula_id = np.squeeze(d[p+42:p+44].view(E+'u2'))
            self.num_value_pairs = np.squeeze(d[p+44:p+46].view(E+'u2'))
    
            # cf: conversion formula
            # 0: parametric, linear
            if self.formula_id==0:
                post = str(2 * 8) + 's'
                fmt_ = 'd' * 2
                cf = ''
                pycode = "raw*param[1]+param[0]" 
    
            # 1: tabular with interpolation
            elif self.formula_id==1:
                post = str(2 * self.num_value_pairs.tolist() * 8) + 's'
                fmt_ = 'd' * 2 * self.num_value_pairs.tolist()
                cf = 'parameter.reshape(' + \
                     str(self.num_value_pairs) + \
                     ', 2).T'
                pycode = "np.interp(raw, param[0,:], param[1,:])"
    
            # 2: tabular
            elif self.formula_id==2:
                post = str(2 * self.num_value_pairs * 8) + 's'
                fmt_ = 'd' * 2 * self.num_value_pairs.tolist()
                cf = 'parameter.reshape(' + \
                     str(self.num_value_pairs) + \
                     ', 2).T'
                pycode = "interpolate.interp1d(param[0,:], param[1,:], kind='nearest')(raw)"
    
            # 6: polynomial function
            elif self.formula_id==6:
                post = str(6 * 8) + 's'
                fmt_ = 'd' * 6
                cf = ''
                pycode = "(param[1]-param[3]*(raw-param[4]-param[5]))/(param[2]*(raw-param[4]-param[5])-param[0])"
    
            # 7: exponential function
            elif self.formula_id==7:
                post = str(7 * 8) + 's'
                fmt_ = 'd' * 7
                cf = ''
                pycode = ''
                print("7: exponential function is not implemented.")
    
            # 8: logarithmic function
            elif self.formula_id==8:
                post = str(7 * 8) + 's'
                fmt_ = 'd' * 7
                cf = ''
                pycode = ''
                print("8: exponential function is not implemented.")
    
            # 9: ASAP2 Rational conversion formula
            elif self.formula_id==9:
                post = str(6 * 8) + 's'
                fmt_ = 'd' * 6
                cf = ''
                pycode = ''
                print("9: ASAP2 Rational conversion formula is not implemented.")
    
            # 10: ASAM-MCD2 Text formula
            elif self.formula_id==10:
                post = '256s'
                fmt_ = '256s'
                cf = ''
                pycode = ''
                print("10: ASAP2 Text formula is not implemented.")
    
            # 11: ASAM-MCD2 Text Table, (COMPU_VTAB)
            elif self.formula_id==11:
                post = str(40 * self.num_value_pairs.tolist()) + 's'
                fmt_ = 'd32s' * self.num_value_pairs.tolist()
                cf = 'dict(zip(np.array(parameters).reshape('+\
                      str(self.num_value_pairs)+\
                     ', 2).T[0,:].astype("float").astype("int").tolist(),np.array(parameters).reshape('+\
                      str(self.num_value_pairs)+\
                     ', 2).T[1,:].tolist()))'
                # pycode = "np.array(list(map(lambda x,par: (par[x]), raw, param)))"
                pycode = "[param[k] for k in np.squeeze(raw).tolist()]"
                # print(pycode)
                # pycode = "np.array(list(map(param, raw)))"
    
                # print("11: ASAM-MCD2 Text Table is not implemented.")
    
            # 12: ASAM-MCD2 Text Range Table (COMPU_VTAB_RANGE)
            elif self.formula_id==12:
                post = str(20 * (self.num_value_pairs.tolist() + 1)) + 's'
                fmt_ = 'ddI' * (self.num_value_pairs.tolist() + 1)
                cf = ''
                print("12: ASAM-MCD2 Text Range Table is not implemented.")
    
            # 132: Date (Based on 7 Byte Date data structure)
            elif self.formula_id==132:
                post = '7s'
                fmt_ = 'H5c'
                cf = ''
                pycode = ''
                print("132: Date is not implemented.")
    
            # 133: time (Based on 6 Byte Time data structure)
            elif self.formula_id==133:
                post = '6s'
                fmt_ = 'IH'
                cf = ''
                pycode = ''
                print("133: time is not implemented.")
    
            elif self.formula_id==65535:
                post = ''
                fmt_ = ''
                cf = ''
                pycode = 'raw'
            else:
                raise ValueError('formula_id: %u does not exist.'%self.formula_id)
    
            fmt = E + '2sHHdd20sHH' + post
            size = calcsize(fmt)
            
            if len(post):
                block_type, \
                block_size, \
                bool_value_range, \
                min_value_range, \
                max_value_range, \
                phy_unit, \
                _, \
                _, \
                parameters = unpack(fmt, d[p:p+size].tobytes())
            else:
                block_type, \
                block_size, \
                bool_value_range, \
                min_value_range, \
                max_value_range, \
                phy_unit, \
                _, \
                _ = unpack(fmt, d[p:p+size].tobytes())
    
            self.block_type = block_type.decode().rstrip('\x00')
            self.block_size = block_size
            self.bool_value_range = bool_value_range
            self.min_value_range = min_value_range
            self.max_value_range = max_value_range
            try:
                self.phy_unit = phy_unit.rstrip('\x00') \
                                        .rsplit(b'\r\n')[0].decode()
            except:
                self.phy_unit = ''
            if len(post):
                parameters = unpack(fmt_, parameters)
            else:
                parameters = ''
            if len(cf):
                parameters = eval(cf)
            if self.formula_id==11:
                pycode = pycode.replace("param", parameters.__repr__())
                
            self.parameters = parameters
            self.pycode = pycode



class  mdfwrite():
    """
    class mdfwrite is so designed, that it will be only invoked by blfload
    internally.
    """
    __BITMATRIX = np.flip(np.arange(64).reshape(8, 8), 1).reshape(64,)
    __CG_BLOCK_SIZE = 26
    __CN_BLOCK_SIZE = 228

    def __init__(self, bl=None):
        self.bl = bl


    def write(self):
        # cc
        endian = '>'
        self.dg = []
        self.cg = []
        self.cn = []
        self.cc = []
        self.dt = []
        self.cc_dict = {}

        self.id = self.IDBLOCK(endian)
        self.hd = self.HDBLOCK(endian, self.bl)
        self.p = self.hd.p_this + self.hd.block_size

        # cc
        for canid, msg in self.bl.parser.message.items():
            cc_local_dict = {}
            cc = self.CCBLOCK(endian, None, True)
            cc.p_this = self.p
            self.p += cc.block_size
            self.cc += [cc]
            cc_local_dict['time'] = cc
            for signal, info in msg['signal'].items():
                cc = self.CCBLOCK(endian, info, False)
                cc.p_this = self.p
                self.p += cc.block_size
                self.cc += [cc]
                cc_local_dict[signal] = cc
            self.cc_dict[canid] = cc_local_dict

        # cn
        for canid, msg in self.bl.parser.message.items():
            period = msg['period']
            if period is None:
                period = 0
            cn = self.CNBLOCK(endian, None, True, None, period)
            cn.p_cc_block = self.cc_dict[canid]['time'].p_this
            cn.p_this = self.p
            p_cn_first = cn.p_this
            self.p += cn.block_size
            cn.p_cn_block = self.p
            self.cn += [cn]
            for signal, info in msg['signal'].items():
                cn = self.CNBLOCK(endian, info, False, self.__BITMATRIX, period)
                cn.p_cc_block = self.cc_dict[canid][signal].p_this
                cn.p_this = self.p
                self.p = cn.p_this + cn.block_size
                cn.p_cn_block = self.p
                self.cn += [cn]
            cn.p_cn_block = 0 # no next
            # cg
            cg = self.CGBLOCK(endian, canid, self.bl)
            cg.p_cn_block = p_cn_first
            cg.p_this = self.p
            self.p += cg.block_size
            self.cg += [cg]

        # dg
        for cg in self.cg:
            dg = self.DGBLOCK(endian, cg.canid)
            dg.p_cg_block = cg.p_this
            dg.p_this = self.p
            self.p += dg.block_size
            dg.p_dg_block = self.p
            self.dg += [dg]
        dg.p_dg_block = 0
        self.hd.p_dg_block = self.dg[0].p_this

        # dt
        for dg in self.dg:
            dt = self.DT(endian, dg.canid, self.bl)
            dt.p_this = self.p
            if dt.d is None:
                dg.p_records = 0
            else:
                dg.p_records = self.p
            self.p += dt.block_size
            self.dt += [dt]


        self.build()

    def build(self):
        self.data = np.zeros(self.p)
        # id
        self.data[self.id.p_this:self.id.p_this+self.id.block_size] = \
            self.id.d
        # hd
        self.hd.build()
        self.data[self.hd.p_this:self.hd.p_this+self.hd.block_size] = \
            self.hd.d
        # cc
        for cc in self.cc:
            cc.build()
            self.data[cc.p_this:cc.p_this+cc.block_size] = cc.d
        for cn in self.cn:
            cn.build()
            self.data[cn.p_this:cn.p_this+cn.block_size] = cn.d
        for cg in self.cg:
            cg.build()
            self.data[cg.p_this:cg.p_this+cg.block_size] = cg.d
        for dg in self.dg:
            dg.build()
            self.data[dg.p_this:dg.p_this+dg.block_size] = dg.d
        for dt in self.dt:
            if dt.d is not None:
                self.data[dt.p_this:dt.p_this+dt.block_size] = dt.d


    class IDBLOCK():

        # E stands for endian, 1: big endian, motorola

        def __init__(self, E):
            self.fmt = E + '8s8s8sHHHH32s'

            self.file_identifier = b'MDF     '
            self.format_identifier = b'3.00    '
            self.program_identifier = b'blfpy   '
            self.block_size = 64
    
            if E == '>':
                self.endian = 1
            elif E == '<':
                self.endian = 0
            else:
                raise ValueError

            self.endian = 1
            self.floating_point_format = 0
            self.version = 300
            self.reserved_1 = 0
            self.reserved_2 = b''
            #
            self.p_this = 0

            self.build()


        def build(self):
                d = pack(self.fmt,
                         self.file_identifier,
                         self.format_identifier,
                         self.program_identifier,
                         self.endian,
                         self.floating_point_format,
                         self.version,
                         self.reserved_1,
                         self.reserved_2)
                self.d = np.frombuffer(d, dtype=np.uint8)
                return self.d


    class HDBLOCK():

        def __init__(self, E, bl):
            self.fmt = E + '2sHIIIH10s8s32s32s32s32s'

            self.block_type = b'HD'
            self.block_size = calcsize(self.fmt)
            self.p_dg_block = 0
            self.p_tx_block = 0
            self.p_pr_block = 0
            self.num_dg_blocks = len(bl.data_index[bl.channel])
            #
            pattern = r'(?P<Y>\d+)/' + \
                      r'(?P<m>\d+)/' + \
                      r'(?P<d>\d+)/\s+' + \
                      r'(?P<H>\d+):' + \
                      r'(?P<M>\d+):' + \
                      r'(?P<S>)\d+'
            dt = re.match(pattern, bl.blf_info['mMeasurementStartTime'])\
                .groupdict()
            self.record_date = f"{dt['d']:>02}:" + \
                               f"{dt['m']:>02}:" + \
                               f"{dt['Y']:>04}"
            self.record_time = f"{dt['H']:>02}:" + \
                               f"{dt['M']:>02}:" + \
                               f"{dt['S']:>02}"
            self.author_name = b''
            self.org_dept_name = b''
            self.project_name = b''
            self.subject_name = b''
            #
            self.p_this = 64


        def build(self):
                d = pack(self.fmt,
                         self.block_type,
                         self.block_size,
                         self.p_dg_block,
                         self.p_tx_block,
                         self.p_pr_block,
                         self.num_dg_blocks,
                         self.record_date.encode(),
                         self.record_time.encode(),
                         self.author_name,
                         self.org_dept_name,
                         self.project_name,
                         self.subject_name)
                self.d = np.frombuffer(d, dtype=np.uint8)
                return self.d


    class DGBLOCK():

        def __init__(self, E, canid):
            self.fmt = E + '2sHIIIIHHI'

            self.block_type = b'DG'
            self.block_size = calcsize(self.fmt)
            self.p_dg_block = 0
            self.p_cg_block = 0
            self.reserved_1 = 0
            self.p_records = 0
            self.num_cg_blocks = 1
            self.num_record_ids = 0
            self.reserved_2 = 0
            #
            self.p_this = 0
            self.canid = canid


        def build(self):
                d = pack(self.fmt,
                         self.block_type,
                         self.block_size,
                         self.p_dg_block,
                         self.p_cg_block,
                         self.reserved_1,
                         self.p_records,
                         self.num_cg_blocks,
                         self.num_record_ids,
                         self.reserved_2)
                self.d = np.frombuffer(d, dtype=np.uint8)
                return self.d


    class CGBLOCK():

        def __init__(self, E, canid, bl):
            # TODO: at constructing
            # num_cn_blocks: num of signals
            # num_records: num of samples of one signal
            self.fmt = E + '2sHIIIHHHI'

            self.block_type = b'CG'
            self.block_size = calcsize(self.fmt)
            self.p_cg_block = 0
            self.p_cn_block = 0
            self.p_tx_block = 0
            self.record_id = 0
            self.num_cn_blocks = len(bl.parser.message[canid])
            self.record_size = 16 # 8 for time in float64, 8 for 8 bytes data
            try:
                self.num_records = len(bl.data_index[bl.channel][canid])
            except: # known KeyError if data of canid does not exist
                self.num_records = 0
            #
            self.p_this = 0
            self.canid = canid


        def build(self):
            d = pack(self.fmt,
                     self.block_type,
                     self.block_size,
                     self.p_cg_block,
                     self.p_cn_block,
                     self.p_tx_block,
                     self.record_id,
                     self.num_cn_blocks,
                     self.record_size,
                     self.num_records)
            self.d = np.frombuffer(d, dtype=np.uint8)
            return self.d


    class CNBLOCK():

        def __init__(self, E, info, time_flg, bm, period):
            # bm: BITMATRIX
            self.fmt = E + '2sHIIIIIH32s128sHHHHdddIIH'
            
            self.block_type = b'CN'
            self.block_size = calcsize(self.fmt)
            self.p_cn_block = 0
            self.p_cc_block = 0
            self.reserved_1 = 0
            self.reserved_2 = 0
            self.p_tx_block = 0
            if time_flg:
                self.cn_type = 1 # 0: data, 1: time
            else:
                self.cn_type = 0
            self.signal_description = b''
            if time_flg:
                self.signal_name = b'time'
                self.bit_start = 63 # motorola
                self.bit_length = 64
                self.signal_data_type = 3 # 0:U, 1: S, 2: f32, 3: f64
            else:
                self.signal_name = info['name'].encode()
                self.bit_start = \
                    int(bm[np.argwhere(bm==info['start']) + info['length']-1])+64
                self.bit_length = info['length']
                self.signal_data_type = 0
            self.bool_value_range = 0
            self.min_value_range = 0
            self.max_value_range = 0
            if period!=0:
                self.sample_rate = period/1000 # second
            else:
                self.sample_rate = 0
            self.p_unique_name = 0
            self.p_tx_block_1 = 0
            self.byte_offset = 0
            #
            self.p_this = 0


        def build(self):
            d = pack(self.fmt,
                     self.block_type,
                     self.block_size,
                     self.p_cn_block,
                     self.p_cc_block,
                     self.reserved_1,
                     self.reserved_2,
                     self.p_tx_block,
                     self.cn_type,
                     self.signal_name,
                     self.signal_description,
                     self.bit_start,
                     self.bit_length,
                     self.signal_data_type,
                     self.bool_value_range,
                     self.min_value_range,
                     self.max_value_range,
                     self.sample_rate,
                     self.p_unique_name,
                     self.p_tx_block_1,
                     self.byte_offset)
            self.d = np.frombuffer(d, dtype=np.uint8)
            return self.d


    class CCBLOCK():

        def __init__(self, E, info, time_flg):
            self.block_type = b'CC'
            if time_flg:
                self.bool_value_range = 0
                self.min_value_range = 0
                self.max_value_range = 0
                self.phy_unit = b's'
                formula_id = 0
            else:
                self.bool_value_range = 1
                self.min_value_range = info['phymin']
                self.max_value_range = info['phymax']
                self.phy_unit = info['unit'].encode()
                if 'enum' in info.keys():
                    formula_id = 11
                else:
                    formula_id = 0

            self.formula_id = formula_id
            if formula_id==0:
                self.num_value_pairs = 0
                self.parameters = [0.0, 1.0]
                self.post = 'dd'
            elif formula_id==11:
                if len(info['enum']):
                    self.parameters = []
                    self.num_value_pairs = len(info['enum'])
                    for k,v in info['enum'].items():
                        self.parameters += [float(k), v.encode()]
                    self.post = 'd32s' * self.num_value_pairs
                else:
                    raise \
                        ValueError(f"\"enums\" is empty.")
            else:
                raise \
                    ValueError(f"formula_id \"{formula_id}\" is not supported.")
            self.E = E
            self.fmt = self.E + '2sHHdd20sHH' + self.post
            self.block_size = calcsize(self.fmt)
            self.build()


        def build(self):
            d = pack(self.fmt,
                     self.block_type,
                     self.block_size,
                     self.bool_value_range,
                     self.min_value_range,
                     self.max_value_range,
                     self.phy_unit,
                     self.formula_id,
                     self.num_value_pairs,
                     *self.parameters)
            self.d = np.frombuffer(d, dtype=np.uint8)
            return self.d


    class DT():

        def __init__(self, E, canid, bl):
            self.p_this = 0
            data_index = bl.data_index[bl.channel]
            try:
                idx = data_index[canid]
            except:
                self.d = None
                self.block_size = 0
                return None
            time = bl.raw_data[3][idx]
            time = time.reshape(time.shape[0], 1).view(E+'u1')
            bb = bl.raw_data[0][idx].astype(np.uint8)
            data = np.concatenate((time, bb), axis=1)
            data = data.reshape(1, data.size)
            self.d = data
            self.block_size = data.size


if __name__ == "__main__":
    # read
    mdf_file = r'../test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
    # np.byte is an alias of np.int8, shall use np.uint8 instead
    # data = np.fromfile(mdf_file, dtype=np.uint8)
    m = mdfread(mdf=mdf_file)
    m.read()


    # write
    # from dbcparser import dbc2code
    # dbc = dbc2code(fn="../test/dbc/IC321_PTCAN_CMatrix_V1.7_PT装车_VBU.dbc")
    # dbc.get_parser()
    # w = mdfwrite()
    # w.write(dbc)