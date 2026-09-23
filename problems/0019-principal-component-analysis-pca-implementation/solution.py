import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Standardize the dataset
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized = (data - mean) / std

    # Compute covariance matrix
    cov_matrix = np.cov(standardized, rowvar=False)

    # Eigendecomposition of symmetric covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]

    # Select top k principal components
    components = eigenvectors[:, :k]

    # Fix eigenvector signs
    for j in range(k):
        for i in range(components.shape[0]):
            if abs(components[i, j]) > 1e-10:
                if components[i, j] < 0:
                    components[:, j] *= -1
                break

    # Round to 4 decimals
    return np.round(components, 4)