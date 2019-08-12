

from itertools import *
from functools import partial
import numpy as np
from numpy_bool_argmax_ext import reversed_bool_argmax as argmax

import unittest
from unittest import TestCase


class TestExtension(TestCase):
    '''
    Multiple tests for the C extension
    '''
    def test_reversed_bool_argmax_type_checking(self):
        # reversed_bool_argmax raises ValueError if no arguments specified
        self.assertRaises(ValueError, argmax)

        # raises ValueError if more than 1 argument indicated
        self.assertRaises(ValueError, argmax, 1, 2)

        # raises ValueError if the given argument has more than 1 dimension
        self.assertRaises(ValueError, argmax, np.zeros([2, 2]))
        self.assertRaises(ValueError, argmax, np.eye(2))
        self.assertRaises(ValueError, argmax, np.ones([2, 2, 2]).astype(np.bool))

        # raises ValueError if the array dtype is not bool
        for dtype in (np.float32, np.int8, np.uint8, np.int32, np.uint32):
            self.assertRaises(ValueError, argmax, np.zeros([2]).astype(dtype))


    def test_reversed_bool_argmax(self):
        # reversed_bool_argmax(a) will return the same as len(a) - np.argmax(a) - 1 if
        # a is a non empty 1D boolean array
        for n, k in product(range(1, 50), range(10)):
            a = np.random.randint(0, 1, n, np.bool)
            self.assertEqual(n-1-argmax(a), np.argmax(a))

            a = np.zeros([n]).astype(np.bool)
            self.assertEqual(n-1-argmax(a), np.argmax(a))

            a = np.ones([n]).astype(np.bool)
            self.assertEqual(n-1-argmax(a), np.argmax(a))

        # reversed_bool_argmax(a) returns 0 if a is a boolean scalar
        self.assertEqual(argmax(False), 0)
        self.assertEqual(argmax(True), 0)


if __name__ == '__main__':
    unittest.main()
