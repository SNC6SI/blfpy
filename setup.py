from setuptools import setup, Extension
from distutils import sysconfig
import os


blfc = Extension(name = 'blfpy.blfc',
                 sources = [os.path.join(os.getcwd(), 'src', 'blfc.c')],
                 include_dirs = [os.path.join(sysconfig.get_python_lib(),
                                              'numpy', 'core', 'include')],
                 library_dirs = [os.path.join(os.getcwd(), 'src')],
                 libraries = ['binlog'])

setup (name = 'blfpy',
       version = '0.2.0',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       maintainer = 'Shen, Chenghao',
       maintainer_email = 'snc6si@gmail.com',
       url = '',
       description = 'blfpy c-extension for python',
       packages=['blfpy'],
       package_data = {'blfpy':['*.dll', '*.pyd']},
       install_requires = ['numpy', 'scipy'],
       ext_modules = [blfc],)
       # script_args = ['build_ext --inplace', 'sdist', 'bdist_wheel'])
