import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)

    standardized_data = (data - mean) / std
    normalized_data = (data - min_data) / (max_data - min_data)

    return standardized_data, normalized_data