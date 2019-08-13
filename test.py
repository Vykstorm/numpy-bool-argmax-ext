

from itertools import *
from functools import partial
import numpy as np
from numpy_bool_argmax_ext import bool_argmax
from argmaxext import argmax

import unittest
from unittest import TestCase


class TestExtension(TestCase):
    '''
    Multiple tests for the C extension
    '''
    def test_bool_argmax_type_checking(self):
        # raises ValueError if no arguments specified
        self.assertRaises(ValueError, bool_argmax)

        # raises ValueError if more than 2 arguments indicated
        self.assertRaises(ValueError, bool_argmax, 1, 2, 3)

        # raises ValueError if the given array has more than 1 dimension
        self.assertRaises(ValueError, bool_argmax, np.zeros([2, 2]))
        self.assertRaises(ValueError, bool_argmax, np.eye(2))
        self.assertRaises(ValueError, bool_argmax, np.ones([2, 2, 2]).astype(np.bool))

        # raises ValueError if the array dtype is not bool
        for dtype in (np.float32, np.int8, np.uint8, np.int32, np.uint32):
            self.assertRaises(ValueError, bool_argmax, np.zeros([2]).astype(dtype))

        # 2nd argument must be a non zero integer value
        self.assertRaises(ValueError, bool_argmax, np.zeros([2], np.bool), 0)


    def test_bool_argmax(self):
        for n, s, k, b in product(range(1, 50), range(1, 15), range(10), range(1, 3)):
            a = np.random.randint(0, b, n, np.bool)

            # bool_argmax(a, stride) return a valid index for any stride value s
            self.assertIn(bool_argmax(a), range(n))

            # a[bool_argmax(a, s)] == True if and only if any(a) == True for any stride value s
            self.assertFalse(a[bool_argmax(a)] ^ np.any(a))

            # bool_argmax(a, s) == 0 if s >= n
            # bool_argmax(a, -s) == n-1 if s >= n
            if s >= n:
                self.assertEqual(bool_argmax(a, s), 0)
                self.assertEqual(bool_argmax(a, -s), n-1)

            # any(a[:bool_argmax(a, s):s]) == False if s > 0
            # any(a[n-1:bool_argmax(a, s):s]) == False if s < 0
            self.assertFalse(any(a[:bool_argmax(a, s):s]))
            self.assertFalse(any(a[n-1:bool_argmax(a, s):s]))


    def test_argmax(self):
        for n, s, k, b in product(range(1, 50), range(1, 5), range(5), range(1, 3)):
            a = np.random.randint(0, b, n, np.bool)

            # argmax(a[::s]) == np.argmax(a[::s])
            # argmax(a[::-s]) == np.argmax(a[::-s])
            self.assertEqual(argmax(a[::s]), np.argmax(a[::s]))
            self.assertEqual(argmax(a[::-s]), np.argmax(a[::-s]))



if __name__ == '__main__':
    unittest.main()
