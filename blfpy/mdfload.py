# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:49:28 2020

@author: SNC6SI
"""
import numpy as np

class mdfload:
    
    def __init__(self, mdf=None):
        if mdf is not None:
            self.mdf = mdf


    def read(self):
        self.data = np.fromfile(self.mdf, dtype=np.uint8)
        
        self.idblock = IDBLOCK(self.data)
        self.endian = self.idblock.default_byte_order
        
        self.hdblock = HDBLOCK(self.data, self.endian)
        
        self.dgblock = DGBLOCK(self.data, self.endian, self.hdblock.p_dg_block)
        
        self.cgblock = CGBLOCK(self.data, self.endian, self.dgblock.p_cg_block)
        
        self.cnblock = CNBLOCK(self.data, self.endian, self.cgblock.p_cn_block)
        
        self.ccblock = CCBLOCK(self.data, self.endian, self.cnblock.p_cc_block)
    
class IDBLOCK:
    
    def __init__(self, data):
        p = 0
        d = data
        self.file_identifier = d[p+0:p+8].tobytes().decode().strip()
        self.format_identifier = d[p+8:p+16].tobytes().decode().strip()
        self.program_identifier = d[p+16:p+24].tobytes().decode().strip()
        self.default_byte_order = np.squeeze(d[p+24:p+26].view('u2'))
        endian = self.default_byte_order.copy()
        if endian:
            self.default_floating_point_format = np.squeeze(data[p+26:p+28].view('>u2'))
            self.version_number = np.squeeze(data[p+28:p+30].view('>u2'))
            self.code_page_number = np.squeeze(data[p+30:p+32].view('>u2'))
            self.standard_flags = np.squeeze(data[p+60:p+62].view('>u2'))
            self.custom_flags = np.squeeze(data[p+62:p+64].view('>u2'))
        else:
            self.default_floating_point_format = np.squeeze(data[p+26:p+28].view('<u2'))
            self.version_number = np.squeeze(data[p+28:p+30].view('<u2'))
            self.code_page_number = np.squeeze(data[p+30:p+32].view('<u2'))
            self.standard_flags = np.squeeze(data[p+60:p+62].view('<u2'))
            self.custom_flags = np.squeeze(data[p+62:p+64].view('<u2'))
    
    @property
    def endian(self):
        return self.default_byte_order


class HDBLOCK:
    
    def __init__(self, data, endian):
        p = 64
        d = data # +208 for 3.30 and is not implemented
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        
        self.record_date = d[p+18:p+28].tobytes().decode() # DD:MM:YYYY
        self.record_time = d[p+28:p+36].tobytes().decode() # HH:MM:SS
        self.author_name = d[p+36:p+68].tobytes().decode().strip()
        self.org_dept_name = d[p+68:p+100].tobytes().decode().strip()
        self.project_name = d[p+100:p+132].tobytes().decode().strip()
        self.subject_name = d[p+132:p+164].tobytes().decode().strip()
        
        if endian:
            self.block_size = np.squeeze(d[p+2:p+4].view('>u2'))
            self.p_dg_block = np.squeeze(d[p+4:p+8].view('>u4'))
            self.p_tx_block = np.squeeze(d[p+8:p+12].view('>u4'))
            self.p_pr_block = np.squeeze(d[p+12:p+16].view('>u4'))
            self.num_dg_blocks = np.squeeze(d[p+16:p+18].view('>u2'))
            
        else:
            self.block_size = np.squeeze(d[p+2:p+4].view('<u2'))
            self.p_dg_block = np.squeeze(d[p+4:p+8].view('<u4'))
            self.p_tx_block = np.squeeze(d[p+8:p+12].view('<u4'))
            self.p_pr_block = np.squeeze(d[p+12:p+16].view('<u4'))
            self.num_dg_blocks = np.squeeze(d[p+16:p+18].view('<u2'))


class DGBLOCK:
    
    def __init__(self, data, endian, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        
        if endian:
            self.block_size = np.squeeze(d[p+2:p+4].view('>u2'))
            self.p_dg_block = np.squeeze(d[p+4:p+8].view('>u4'))
            self.p_cg_block = np.squeeze(d[p+8:p+12].view('>u4'))
            self.p_d_record = np.squeeze(d[p+16:p+20].view('>u4'))
            self.num_cg_blocks = np.squeeze(d[p+20:p+22].view('>u2'))
            self.num_record_ids = np.squeeze(d[p+22:p+24].view('>u2'))
            
        else:
            self.block_size = np.squeeze(d[p+2:p+4].view('<u2'))
            self.p_dg_block = np.squeeze(d[p+4:p+8].view('<u4'))
            self.p_cg_block = np.squeeze(d[p+8:p+12].view('<u4'))
            self.p_d_record = np.squeeze(d[p+16:p+20].view('<u4'))
            self.num_cg_blocks = np.squeeze(d[p+20:p+22].view('<u2'))
            self.num_record_ids = np.squeeze(d[p+22:p+24].view('<u2'))


class CGBLOCK:
    
    def __init__(self, data, endian, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        
        if endian:
            self.block_size = np.squeeze(d[p+2:p+4].view('>u2'))
            self.p_cg_block = np.squeeze(d[p+4:p+8].view('>u4'))
            self.p_cn_block = np.squeeze(d[p+8:p+12].view('>u4'))
            self.p_tx_block = np.squeeze(d[p+12:p+16].view('>u4'))
            
            self.record_id = np.squeeze(d[p+16:p+18].view('>u2'))
            self.num_channels = np.squeeze(d[p+18:p+20].view('>u2'))
            self.data_record_size = np.squeeze(d[p+20:p+22].view('>u2'))
            self.num_records = np.squeeze(d[p+22:p+26].view('>u4'))
            
        else:
            self.block_size = np.squeeze(d[p+2:p+4].view('<u2'))
            self.p_cg_block = np.squeeze(d[p+4:p+8].view('<u4'))
            self.p_cn_block = np.squeeze(d[p+8:p+12].view('<u4'))
            self.p_tx_block = np.squeeze(d[p+12:p+16].view('<u4'))
            
            self.record_id = np.squeeze(d[p+16:p+18].view('<u2'))
            self.num_channels = np.squeeze(d[p+18:p+20].view('<u2'))
            self.data_record_size = np.squeeze(d[p+20:p+22].view('<u2'))
            self.num_records = np.squeeze(d[p+22:p+26].view('<u4'))


class CNBLOCK:

    def __init__(self, data, endian, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        self.signal_name = d[p+26:p+58].tobytes().decode().strip()
        self.signal_description = d[p+58:p+186].tobytes().decode().strip()
        
        if endian:
            self.block_size = np.squeeze(d[p+2:p+4].view('>u2'))
            self.p_cn_block = np.squeeze(d[p+4:p+8].view('>u4'))
            self.p_cc_block = np.squeeze(d[p+8:p+12].view('>u4'))
            self.p_tx_block = np.squeeze(d[p+20:p+24].view('>u4'))
            # channel_type: 0=data, 1=time
            self.channel_type = np.squeeze(d[p+24:p+26].view('>u2'))
            self.bit_pos = np.squeeze(d[p+186:p+188].view('>u2'))
            self.num_bits = np.squeeze(d[p+188:p+190].view('>u2'))
            self.signal_data_type = np.squeeze(d[p+190:p+192].view('>u2'))
            self.bool_value_range = np.squeeze(d[p+192:p+194].view('>u2'))
            # if bool==false, maybe not to implement these value ranges
            self.min_value_range = np.squeeze(d[p+194:p+202].view('>f8'))
            self.max_value_range = np.squeeze(d[p+202:p+210].view('>f8'))
            self.sample_rate = np.squeeze(d[p+210:p+218].view('>f8'))
            self.p_unique_name = np.squeeze(d[p+218:p+222].view('>u4'))
            self.byte_offset = np.squeeze(d[p+226:p+228].view('>u2'))
        else:
            self.block_size = np.squeeze(d[p+2:p+4].view('<u2'))
            self.p_cn_block = np.squeeze(d[p+4:p+8].view('<u4'))
            self.p_cc_block = np.squeeze(d[p+8:p+12].view('<u4'))
            self.p_tx_block = np.squeeze(d[p+20:p+24].view('<u4'))
            # channel_type: 0=data, 1=time
            self.channel_type = np.squeeze(d[p+24:p+26].view('<u2'))
            self.bit_pos = np.squeeze(d[p+186:p+188].view('<u2'))
            self.num_bits = np.squeeze(d[p+188:p+190].view('<u2'))
            self.signal_data_type = np.squeeze(d[p+190:p+192].view('<u2'))
            self.bool_value_range = np.squeeze(d[p+192:p+194].view('<u2'))
            self.min_value_range = np.squeeze(d[p+194:p+202].view('<f8'))
            self.max_value_range = np.squeeze(d[p+202:p+210].view('<f8'))
            self.sample_rate = np.squeeze(d[p+210:p+218].view('<f8'))
            self.p_unique_name = np.squeeze(d[p+218:p+222].view('<u4'))
            self.byte_offset = np.squeeze(d[p+226:p+228].view('<u2'))
            
            
class CCBLOCK:
    
    def __init__(self, data, endian, p):
        d = data
        
        self.block_type = d[p+0:p+2].tobytes().decode().strip()
        self.phy_unit = d[p+22:p+42].tobytes().decode().strip()
        
        if endian:
            self.block_size = np.squeeze(d[p+2:p+4].view('>u2'))
            self.bool_value_range = np.squeeze(d[p+4:p+6].view('>u2'))
            self.min_value_range = np.squeeze(d[p+6:p+14].view('>f8'))
            self.max_value_range = np.squeeze(d[p+14:p+22].view('>f8'))
            self.formula_id = np.squeeze(d[p+42:p+44].view('>u2'))
            self.size_info = np.squeeze(d[p+44:p+46].view('>u2'))
            if self.formula_id==0:
                self.parameters = np.squeeze(d[p+46:p+62].view('>f8'))
            elif self.formula_id==6 or self.formula_id==9:
                self.parameters = np.squeeze(d[p+46:p+94].view('>f8'))
            elif self.formula_id==7 or self.formula_id==8:
                self.parameters = np.squeeze(d[p+46:p+102].view('>f8'))
            
        else:
            self.block_size = np.squeeze(d[p+2:p+4].view('<u2'))
            self.bool_value_range = np.squeeze(d[p+4:p+6].view('<u2'))
            self.min_value_range = np.squeeze(d[p+6:p+14].view('<f8'))
            self.max_value_range = np.squeeze(d[p+14:p+22].view('<f8'))
            self.formula_id = np.squeeze(d[p+42:p+44].view('<u2'))
            self.size_info = np.squeeze(d[p+44:p+46].view('<u2'))
            if self.formula_id==0:
                self.parameters = np.squeeze(d[p+46:p+62].view('<f8'))
            elif self.formula_id==6 or self.formula_id==9:
                self.parameters = np.squeeze(d[p+46:p+94].view('<f8'))
            elif self.formula_id==7 or self.formula_id==8:
                self.parameters = np.squeeze(d[p+46:p+102].view('<f8'))



if __name__ == "__main__":
    mdf_file = r'../test/2020-07-17_19_IC321_HEV150_SW2.2.4_C2.2.1_FCH_NoreqI_01.dat'
    # np.byte is an alias of np.int8, shall use np.uint8 instead
    # data = np.fromfile(mdf_file, dtype=np.uint8)
    m = mdfload(mdf=mdf_file)
    m.read()