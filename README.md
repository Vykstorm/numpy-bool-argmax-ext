
## Introduction

This library provides an alternative to ```numpy.argmax``` to get the maximum value in a 1D boolean array with inverted stride (-1), without performing a copy of the input array, and thus increasing the performance


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

The ```argmax``` will perform additional optimizations compared to ```np.argmax``` when the input argument is a 1D boolean array with inverted stride (-1). <br/>
Execute the next benchmark:

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
