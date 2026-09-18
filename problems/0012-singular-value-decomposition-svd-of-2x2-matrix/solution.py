import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:

    B = A @ A.T

    a = B[0, 0]
    b = B[0, 1]
    d = B[1, 1]

    theta = 0.5 * np.arctan2(2 * b, a - d)

    c = np.cos(theta)
    s = np.sin(theta)

    U = np.array([
        [c, -s],
        [s,  c]
    ])

    D = U.T @ B @ U

    eigvals = np.diag(D)
    S = np.sqrt(np.maximum(eigvals, 0))

    V = A.T @ U @ np.diag(1 / S)
    Vt = V.T

    return U, S, Vt