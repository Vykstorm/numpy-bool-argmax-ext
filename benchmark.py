

from timeit import timeit
from argmaxext import argmax
import numpy as np



if __name__ == '__main__':
    # Build m 1D arrays of size n with a random item set to True (the rest is False)
    n, m = 2 ** 15, 2 ** 12
    arrays = []
    for i in range(m):
        a = np.zeros([n], np.bool)
        a[np.random.randint(0, n, 1, np.uint32)] = True
        arrays.append(a)

    print(f"Running benchmark over {m} boolean arrays of size {arrays[0].shape}...\n\n")

    # Compare np.argmax(a[::-1]) with reversed_bool_argmax(a)
    def foo():
        for a in arrays:
            np.argmax(a[::-1])

    def bar():
        for a in arrays:
            argmax(a)

    def qux():
        for a in arrays:
            np.argmax(a)


    k = 100
    print("Running np.argmax(a)...")
    time = timeit(qux, number=k)
    print("-> Average time: {:6f} msecs".format(1000 * time / (k * m)))
    print()

    print("Running np.argmax(a[::-1])...")
    time = timeit(foo, number=k)
    print("-> Average time: {:6f} msecs".format(1000 * time / (k * m)))
    print()

    print("-> Running reversed_bool_argmax(a)...")
    time = timeit(bar, number=k)
    print("Average time: {:6f} msecs".format(1000 * time / (k * m)))
    print()
