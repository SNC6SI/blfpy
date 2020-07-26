# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:49:28 2020

@author: SNC6SI: Shen,Chenghao <snc6si@gmail.com>
"""

from struct import unpack, calcsize
import math
import numpy as np

class mdfload:
    
    def __init__(self, mdf=None):
        if mdf is not None:
            self.mdf = mdf


    def read(self):
        self.data = np.fromfile(self.mdf, dtype=np.uint8)
        
        self.idblock = IDBLOCK(self.data)
        if self.idblock.endian:
            self.endian = '>'
        else:
            self.endian = '<'
        
        self.hdblock = HDBLOCK(self.data, self.endian)


        # DG
        if self.hdblock.num_dg_blocks:
            self.dgblocks = []
            dgblock = DGBLOCK(self.data, self.endian, self.hdblock.p_dg_block)
            self.dgblocks += [dgblock]
            while dgblock.p_dg_block:
                dgblock = DGBLOCK(self.data, self.endian, dgblock.p_dg_block)
                self.dgblocks += [dgblock]
        
        # print("num_dg_blocks in hdblock: %u\nnum_dg_blocks read: %u" % \
        #        (self.hdblock.num_dg_blocks, len(self.dgblocks)))
            
        # CG
        for dgblock in self.dgblocks:
            if dgblock.num_cg_blocks:
                cgblocks = []
                cgblock = CGBLOCK(self.data, self.endian, dgblock.p_cg_block)
                cgblocks += [cgblock]
                while cgblock.p_cg_block:
                    cgblock = CGBLOCK(self.data, self.endian, cgblock.p_cg_block)
                    cgblocks += [cgblock]
                dgblock.cgblocks = cgblocks


        # CN
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                cnblocks = []
                cnblock = CNBLOCK(self.data, self.endian, cgblock.p_cn_block)
                cnblocks += [cnblock]
                while cnblock.p_cn_block:
                    cnblock = CNBLOCK(self.data, self.endian, cnblock.p_cn_block)
                    cnblocks += [cnblock]
                cgblock.cnblocks = cnblocks


        # DR
        for dgblock in self.dgblocks:
            for cgblock in dgblock.cgblocks:
                if dgblock.num_record_ids==0:
                    dgblock.records = DR(self.data,
                                              dgblock.num_record_ids,
                                              dgblock.p_records,
                                              cgblock.record_size,
                                              cgblock.num_records)


        # CC
        for dgblock in self.dgblocks:
             for cgblock in dgblock.cgblocks:
                 for cnblock in cgblock.cnblocks:
                     # print(cnblock.signal_name)
                     cnblock.ccblock = \
                         CCBLOCK(self.data, self.endian, cnblock.p_cc_block)


        # raw
        for dgblock in self.dgblocks:
            bb = dgblock.records.mat
            for cgblock in dgblock.cgblocks:
                for cnblock in cgblock.cnblocks:
                    byte_start = cnblock.byte_offset + \
                                 math.floor(cnblock.bit_start/8)
                    byte_end = cnblock.byte_offset + \
                        math.ceil((cnblock.bit_start + cnblock.bit_length)/8)
                    byte_length = byte_end-byte_start
                    if byte_length<=8:
                        view = self.endian + 'u' + str(2**(math.ceil(math.log2(byte_length))))
                        raw = bb[:, byte_start:byte_end].copy().view(view)
                        raw = (raw>>(cnblock.bit_start%8))&np.uint64((2**cnblock.bit_length-1))
                    else:
                        raw = bb[:, byte_start:byte_end].copy()
                    cnblock.raw = raw


        # phy

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
class DR:
    
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

        # 0: parametric, linear
        if self.formula_id==0:
            post = str(2 * 8) + 's'
            fmt_ = 'd' * 2

        # 1: tabular with interpolation
        elif self.formula_id==1:
            post = str(2 * self.num_value_pairs.tolist() * 8) + 's'
            fmt_ = 'd' * 2 * self.num_value_pairs.tolist()

        # 2: tabular
        elif self.formula_id==2:
            post = str(2 * self.num_value_pairs * 8) + 's'
            fmt_ = 'd' * 2 * self.num_value_pairs.tolist()

        # 6: polynomial function
        elif self.formula_id==6:
            post = str(6 * 8) + 's'
            fmt_ = 'd' * 6

        # 7: exponential function
        elif self.formula_id==7:
            post = str(7 * 8) + 's'
            fmt_ = 'd' * 7

        # 8: logarithmic function
        elif self.formula_id==8:
            post = str(7 * 8) + 's'
            fmt_ = 'd' * 7

        # 9: ASAP2 Rational conversion formula
        elif self.formula_id==9:
            post = str(6 * 8) + 's'
            fmt_ = 'd' * 6
            
        # 10: ASAM-MCD2 Text formula
        elif self.formula_id==10:
            post = '256s'
            fmt_ = '256s'

        # 11: ASAM-MCD2 Text Table, (COMPU_VTAB)
        elif self.formula_id==11:
            post = str(40 * self.num_value_pairs.tolist()) + 's'
            fmt_ = 'd32s' * self.num_value_pairs.tolist()

        # 12: ASAM-MCD2 Text Range Table (COMPU_VTAB_RANGE)
        elif self.formula_id==12:
            post = str(20 * (self.num_value_pairs.tolist() + 1)) + 's'
            fmt_ = 'ddI' * (self.num_value_pairs.tolist() + 1)

        # 132: Date (Based on 7 Byte Date data structure)
        elif self.formula_id==132:
            post = '7s'
            fmt_ = 'H5c'

        # 133: time (Based on 6 Byte Time data structure)
        elif self.formula_id==133:
            post = '6s'
            fmt_ = 'IH'

        elif self.formula_id==65535:
            post = ''
            fmt_ = ''
        else:
            raise ValueError('formula_id: %u does not exist.'%self.formula_id)

        # print(post)
        fmt = E + '2sHHdd20sHH' + post
        size = calcsize(fmt)
        # print(fmt)
        # print(size)
        # print(len(post))
        
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
            self.parameters = parameters
        else:
            self.parameters = ''


if __name__ == "__main__":
    mdf_file = r'../test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
    # np.byte is an alias of np.int8, shall use np.uint8 instead
    # data = np.fromfile(mdf_file, dtype=np.uint8)
    m = mdfload(mdf=mdf_file)
    m.read()