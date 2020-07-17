from setuptools import setup, Extension
from distutils import sysconfig
import os


blfc = Extension(name = 'blfpy.blfc',
                 include_dirs = [os.path.join(sysconfig.get_python_lib(),
                                              'numpy', 'core', 'include'),
                                 os.path.join(os.getcwd(), 'src')],
                 library_dirs = [os.path.join(os.getcwd(), 'src')],
                 libraries = ['binlog'],
                 sources = [os.path.join(os.getcwd(),
                                         'src', 'blfc.c')])

setup (name = 'blfpy',
       version = '0.1.0',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       maintainer = 'Shen, Chenghao',
       maintainer_email = 'snc6si@gmail.com',
       url = '',
       description = 'blfpy c-extension for python',
       data_files = [os.path.join(os.getcwd(), 'src', 'binlog.dll')],
       packages=['blfpy'],
       ext_modules = [blfc],)
       # script_args = ['build_ext --inplace', 'sdist', 'bdist_wheel'])
