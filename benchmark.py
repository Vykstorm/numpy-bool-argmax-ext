

from timeit import timeit
from argmaxext import argmax
import numpy as np
from numpy_bool_argmax_ext import bool_argmax


if __name__ == '__main__':
    # Build m 1D arrays of size n with a random item set to True (the rest is False)
    n, m = 2 ** 15, 2 ** 12
    arrays = []
    for i in range(m):
        a = np.zeros([n], np.bool)
        a[np.random.randint(0, n, 1, np.uint32)] = True
        arrays.append(a)

    print(f"Running benchmark over {m} boolean arrays of size {arrays[0].shape}...\n\n")

    # Compare np.argmax(a[::-1]) with bool_argmax(a[::-1])
    def foo():
        for a in arrays:
            np.argmax(a[::-1])

    def bar():
        for a in arrays:
            argmax(a[::-1])

    def qux():
        for a in arrays:
            np.argmax(a)


    k = 100
    print("np.argmax(a)...")
    time = timeit(qux, number=k)
    print("-> Average time: {:6f} msecs".format(1000 * time / (k * m)))
    print()

    for s in (-1, 2, -2, 3, -3):
        print("=" * 30)
        print(f"stride = {s}")

        print(f"-> np.argmax(a[::{s}])...")
        time = timeit(foo, number=k)
        print("-> Average time: {:6f} msecs".format(1000 * time / (k * m)))
        print()

        print(f"-> argmax(a[::{s}])...")
        time = timeit(bar, number=k)
        print("Average time: {:6f} msecs".format(1000 * time / (k * m)))
        print()
