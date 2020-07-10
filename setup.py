from distutils.core import setup, Extension
from distutils import sysconfig
import os

mdl_blfpy = Extension('blfpy',
                      include_dirs = [os.path.join(sysconfig.get_python_lib(),
                                                  'numpy', 'core', 'include')],
                      libraries = ['binlog', 'python3'],
                      sources = ['blfpy.c'])

setup (name = 'blfpy',
       version = '0.1',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       description = 'blfpy c-extension for python',
       data_files = ["binlog.dll"],
       ext_modules = [mdl_blfpy])
