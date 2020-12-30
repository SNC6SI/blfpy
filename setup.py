# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:16:44 2020

@author: SNC6SI: Shen, Chenghao <snc6si@gmail.com>
"""

from setuptools import setup, Extension
from distutils import sysconfig
from glob import glob
import os


blfc = Extension(name = 'blfpy.blfc',
                 sources = ['src/blfc2.c'] + glob('src/zlib/*.c'),
                 include_dirs = [os.path.join(sysconfig.get_python_lib(),
                                              'numpy', 'core', 'include'),
                                 'src'])

setup (name = 'blfpy',
       version = '0.5.1',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       maintainer = 'Shen, Chenghao',
       maintainer_email = 'snc6si@gmail.com',
       url = '',
       description = 'blfpy c-extension for python',
       license = "MIT",
       packages = ['blfpy'],
       package_data = {'blfpy':['*.dll', '*.pyd']},
       install_requires = ['numpy', 'scipy'],
       ext_modules = [blfc],)
       # script_args = ['build_ext --inplace', 'sdist', 'bdist_wheel'])
