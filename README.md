
## Introduction

This library provides an alternative to ```numpy.argmax``` to get the maximum value in a 1D boolean array with strides, without performing a copy of the input array, and thus increasing the performance.

The rationale is explained in this <a href="https://stackoverflow.com/questions/57346182/why-does-numpy-not-short-circuit-on-non-contiguous-arrays/57348030#57348030">stackoverflow question</a>


## Requirements

The only dependency is numpy.

For now, only tested with Python>=3.7 and numpy==1.16.4

## Installation

Via ```setup.py``` script:

```bash
git clone https://github.com/Vykstorm/numpy-bool-argmax-ext.git
cd numpy-bool-argmax-ext
python setup.py install
```

## Usage

Now you only need to use the function ```argmax``` defined in this library instead of ```np.argmax```

e.g:
```python
import numpy as np
from argmaxext import argmax

a = np.random.randint(0, 2, 10000, np.bool)
print(argmax(a))
```

Execute the next benchmark to compare both functions when dealing with boolean 1D arrays and -1 as stride value for example:

```python
from timeit import timeit

a = np.zeros([2 ** 18], np.bool)
# Worst case scenario (only the first item is True)
a[0] = True

k = 10000
print("np.argmax(a[::-1]) average time: {:6f} msecs".format(
    1000*timeit(lambda: np.argmax(a[::-1]), number=k) / k
))

print("argmax(a[::-1]) average time: {:6f} msecs".format(
    1000*timeit(lambda: argmax(a[::-1]), number=k) / k
))
```
