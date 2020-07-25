# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:49:28 2020

@author: SNC6SI: Shen,Chenghao <snc6si@gmail.com>
"""
import numpy as np

class mdfload:
    
    def __init__(self, mdf=None):
        if mdf is not None:
            self.mdf = mdf


    def read(self):
        self.data = np.fromfile(self.mdf, dtype=np.uint8)
        
        self.idblock = IDBLOCK(self.data)
        if self.idblock.default_byte_order:
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
                    dgblock.data_records = DR(self.data,
                                              dgblock.num_record_ids,
                                              dgblock.p_d_record,
                                              cgblock.record_size,
                                              cgblock.num_records)
                

        # CC
        for dgblock in self.dgblocks:
             for cgblock in dgblock.cgblocks:
                 for cnblock in cgblock.cnblocks:
                     cnblock.ccblock = \
                         CCBLOCK(self.data, self.endian, cnblock.p_cc_block)
                 


class IDBLOCK:
    """
    useful infos:
        - version
    """
    
    def __init__(self, data):
        p = 0
        d = data
        self.file_identifier = d[p+0:p+8].tobytes().decode().strip()
        self.format_identifier = d[p+8:p+16].tobytes().decode().strip()
        self.program_identifier = d[p+16:p+24].tobytes().decode().strip()
        self.default_byte_order = np.squeeze(d[p+24:p+26].view('u2'))
        if self.default_byte_order:
            E = '>'
        else:
            E = '<'

        self.default_floating_point_format = np.squeeze(data[p+26:p+28].view(E+'u2'))
        self.version = np.squeeze(data[p+28:p+30].view(E+'u2'))
        self.code_page_number = np.squeeze(data[p+30:p+32].view(E+'u2'))
        self.standard_flags = np.squeeze(data[p+60:p+62].view(E+'u2'))
        self.custom_flags = np.squeeze(data[p+62:p+64].view(E+'u2'))

    
    @property
    def endian(self):
        return self.default_byte_order


class HDBLOCK:
    """
    useful infos:
        - p_dg_block
        - num_dg_blocks
    """
    
    def __init__(self, data, E):
        p = 64
        d = data # +208 for 3.30 and is not implemented
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        
        self.record_date = d[p+18:p+28].tobytes().decode() # DD:MM:YYYY
        self.record_time = d[p+28:p+36].tobytes().decode() # HH:MM:SS
        self.author_name = d[p+36:p+68].tobytes().decode().strip()
        self.org_dept_name = d[p+68:p+100].tobytes().decode().strip()
        self.project_name = d[p+100:p+132].tobytes().decode().strip()
        self.subject_name = d[p+132:p+164].tobytes().decode().strip()
        
        self.block_size = np.squeeze(d[p+2:p+4].view(E+'u2'))
        self.p_dg_block = np.squeeze(d[p+4:p+8].view(E+'u4'))
        self.p_tx_block = np.squeeze(d[p+8:p+12].view(E+'u4'))
        self.p_pr_block = np.squeeze(d[p+12:p+16].view(E+'u4'))
        self.num_dg_blocks = np.squeeze(d[p+16:p+18].view(E+'u2'))


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
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()

        self.block_size = np.squeeze(d[p+2:p+4].view(E+'u2'))
        self.p_dg_block = np.squeeze(d[p+4:p+8].view(E+'u4'))
        self.p_cg_block = np.squeeze(d[p+8:p+12].view(E+'u4'))
        self.p_d_record = np.squeeze(d[p+16:p+20].view(E+'u4'))
        self.num_cg_blocks = np.squeeze(d[p+20:p+22].view(E+'u2'))
        self.num_record_ids = np.squeeze(d[p+22:p+24].view(E+'u2'))


# DATARECORDS
class DR:
    
    def __init__(self, data, record_id, p, s, n):
        # p: p_d_record
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
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()

        self.block_size = np.squeeze(d[p+2:p+4].view(E+'u2'))
        self.p_cg_block = np.squeeze(d[p+4:p+8].view(E+'u4'))
        self.p_cn_block = np.squeeze(d[p+8:p+12].view(E+'u4'))
        self.p_tx_block = np.squeeze(d[p+12:p+16].view(E+'u4'))
        
        self.record_id = np.squeeze(d[p+16:p+18].view(E+'u2'))
        self.num_cn_blocks = np.squeeze(d[p+18:p+20].view(E+'u2'))
        self.record_size = np.squeeze(d[p+20:p+22].view(E+'u2'))
        self.num_records = np.squeeze(d[p+22:p+26].view(E+'u4'))


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
        
        - bit_start_pos
        - bit_length
        - byte_offset
    """

    def __init__(self, data, E, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        self.signal_name = d[p+26:p+58].tobytes().decode().strip()
        try:
            self.signal_description = d[p+58:p+186].tobytes()\
                        .rstrip('\x00')\
                        .rsplit(b'\r\n')[0]\
                        .decode(encoding='utf-8').strip()
        except:
            self.signal_description = ''
            
        self.block_size = np.squeeze(d[p+2:p+4].view(E+'u2'))
        self.p_cn_block = np.squeeze(d[p+4:p+8].view(E+'u4'))
        self.p_cc_block = np.squeeze(d[p+8:p+12].view(E+'u4'))
        self.p_tx_block = np.squeeze(d[p+20:p+24].view(E+'u4'))
        # cn_type: 0=data, 1=time
        self.cn_type = np.squeeze(d[p+24:p+26].view(E+'u2'))
        self.bit_start_pos = np.squeeze(d[p+186:p+188].view(E+'u2'))
        self.bit_length = np.squeeze(d[p+188:p+190].view(E+'u2'))
        self.signal_data_type = np.squeeze(d[p+190:p+192].view(E+'u2'))
        self.bool_value_range = np.squeeze(d[p+192:p+194].view(E+'u2'))
        # if bool==false, maybe not to implement these value ranges
        self.min_value_range = np.squeeze(d[p+194:p+202].view(E+'f8'))
        self.max_value_range = np.squeeze(d[p+202:p+210].view(E+'f8'))
        self.sample_rate = np.squeeze(d[p+210:p+218].view(E+'f8'))
        self.p_unique_name = np.squeeze(d[p+218:p+222].view(E+'u4'))
        self.byte_offset = np.squeeze(d[p+226:p+228].view(E+'u2'))

            
            
class CCBLOCK:
    """
    useful infos:
        - phy_unit
        - formula_id
        - parameters
    """
    
    def __init__(self, data, E, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        try:
            self.phy_unit = d[p+22:p+42].tobytes()\
                    .rstrip('\x00')\
                    .rsplit(b'\r\n')[0]\
                    .decode().strip()
        except:
            self.phy_unit = ''

        self.block_size = np.squeeze(d[p+2:p+4].view(E+'u2'))
        self.bool_value_range = np.squeeze(d[p+4:p+6].view(E+'u2'))
        self.min_value_range = np.squeeze(d[p+6:p+14].view(E+'f8'))
        self.max_value_range = np.squeeze(d[p+14:p+22].view(E+'f8'))
        self.formula_id = np.squeeze(d[p+42:p+44].view(E+'u2'))
        self.size_info = np.squeeze(d[p+44:p+46].view(E+'u2'))
        if self.formula_id==0:
            self.parameters = np.squeeze(d[p+46:p+62].view(E+'f8'))
        elif self.formula_id==6 or self.formula_id==9:
            self.parameters = np.squeeze(d[p+46:p+94].view(E+'f8'))
        elif self.formula_id==7 or self.formula_id==8:
            self.parameters = np.squeeze(d[p+46:p+102].view(E+'f8'))


if __name__ == "__main__":
    mdf_file = r'../test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
    # np.byte is an alias of np.int8, shall use np.uint8 instead
    # data = np.fromfile(mdf_file, dtype=np.uint8)
    m = mdfload(mdf=mdf_file)
    m.read()