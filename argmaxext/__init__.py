

import numpy as np
from numpy_bool_argmax_ext import bool_argmax
from functools import partial

__all__ = ['argmax']


def argmax(x, axis=None, out=None):
    '''
    This function is equivalent to numpy.argmax, but it is optimized
    when you pass a 1D boolean array with strides and the rest of the arguments
    (axis and out) are set to None.
    '''
    x = np.asarray(x)
    default_argmax = partial(np.argmax, x, axis, out)

    if x.dtype != np.bool or x.ndim != 1 or (x.flags.aligned and x.flags.c_contiguous) or x.base is None:
        return default_argmax()

    base, stride = x.base, x.strides[0]
    if (base.size-1)//abs(stride)+1 != x.size:
        return default_argmax()

    k = bool_argmax(base, stride)
    if stride < 0:
        k = base.size-k-1
    return k//stride
