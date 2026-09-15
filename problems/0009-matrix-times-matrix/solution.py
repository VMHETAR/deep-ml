import numpy as np
def matrixmul(a: list[list[int|float]],
              b: list[list[int|float]]) -> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    c = np.matmul(a, b)
    return c