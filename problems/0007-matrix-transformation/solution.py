import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    return -1 if np.linalg.det(T) == 0 or np.linalg.det(S) == 0 else (np.linalg.inv(T) @ np.array(A) @ np.array(S)).tolist()