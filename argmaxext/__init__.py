

import numpy as np
from numpy_bool_argmax_ext import reversed_bool_argmax

__all__ = ['argmax']


def argmax(x, axis=None, out=None):
    '''
    This function is equivalent to numpy.argmax, but it is optimized
    when you pass a 1D boolean array with strides (-1,) and the rest of the arguments
    (axis and out) are set to None.
    '''
    x = np.asarray(x)

    if x.dtype == np.bool and (x.ndim == 0 or (x.ndim == 1 and x.strides == (-1,))) and x.base is not None:
        return len(x)-reversed_bool_argmax(x.base)-1
    return np.argmax(x, axis, out)
