import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(len(b))
    for iteration in range(n):
        new_x = []
        for i in range(len(A)):
            total = 0
            for j in range(len(A)):
                if i != j:
                    total += A[i][j] * x[j]
            new_value = (b[i] - total) / A[i][i]
            new_x.append(new_value)
        x = new_x
    return np.round(x, 4).tolist()