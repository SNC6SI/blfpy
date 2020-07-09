from distutils.core import setup, Extension
from distutils import sysconfig

module1 = Extension('blfpy',
                    include_dirs = [sysconfig.get_python_lib() + '\\numpy\\core\\include\\'],
                    libraries = ['binlog', 'python3'],
                    sources = ['blfpy.c'])

setup (name = 'blfpy',
       version = '0.1',
       author = 'Shen, Chenghao',
       author_email='snc6si@gmail.com',
       description = 'blfpy in CPython',
       data_files = [(sysconfig.get_python_lib(), ["binlog.dll"])],
       ext_modules = [module1])
