import numpy as np

def superposition_reconstruct(W, b, X):
    """
    Compute reconstructed features for the toy superposition model.

    Args:
        W: array of shape (n_hidden, n_features)
        b: array of shape (n_features,)
        X: array of shape (batch_size, n_features)

    Returns:
        list of lists of shape (batch_size, n_features) with reconstructed features
    """
    W = np.array(W)
    b = np.array(b)
    X = np.array(X)
    
    result = []
    for x in X:
        reconstructed = np.maximum(0, W.T @ W @ x+ b)
        result.append(reconstructed)
    return [r.tolist() for r in result]
