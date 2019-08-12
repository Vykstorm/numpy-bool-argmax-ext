

from distutils.core import setup, Extension
import numpy as np


ext = Extension('numpy_bool_argmax_ext',
    sources = ['argmax.c'],
    include_dirs=[np.get_include()],
    language='c',
    )

setup(
    name = 'numpy-bool-argmax-ext',
    version = '1.0.0',
    description = 'Additional helper methods to improve numpy.argmax performance for non-contiguous 1D boolean arrays',
    ext_modules = [ext])
