import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    cov = np.cov(vectors)
    return cov.tolist()