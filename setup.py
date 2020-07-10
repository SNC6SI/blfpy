from distutils.core import setup, Extension
from distutils import sysconfig
import os

mdl_blfpy = Extension('blfpy',
                      include_dirs = [os.path.join(sysconfig.get_python_lib(),
                                                  'numpy', 'core', 'include'),
                                      os.path.join(os.getcwd(), 'blfpy')],
                      library_dirs = [os.path.join(os.getcwd(), 'blfpy')],
                      libraries = ['binlog'],
                      sources = [os.path.join(os.getcwd(),
                                              'blfpy', 'blfpy.c')])

setup (name = 'blfpy',
       version = '0.1',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       description = 'blfpy c-extension for python',
       data_files = [os.path.join(os.getcwd(), 'blfpy', 'binlog.dll')],
       ext_modules = [mdl_blfpy],
       script_args = ['build', 'sdist', 'bdist'])
