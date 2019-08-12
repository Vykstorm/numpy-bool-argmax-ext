

from distutils.core import setup, Extension
import numpy as np


ext = Extension('numpy_bool_argmax_ext',
    sources =['argmax.c'],
    include_dirs=[np.get_include()],
    language='c',
    )

setup(
    name='numpy-bool-argmax-ext',
    version='1.0.0',
    author='Vykstorm',
    author_email='victorruizgomezdev@gmail.com',

    description='Additional helper methods to improve numpy.argmax performance for 1D boolean arrays with stride -1',
    ext_modules=[ext],
    package_dir={'argmaxext': 'argmaxext'})
