import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """

    w = np.zeros(X_train.shape[1])
    b = 0.0

    learning_rate = 0.01
    iterations = 10000

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for i in range(iterations):

        # Forward pass
        z = X_train @ w + b
        y_hat = sigmoid(z)

        # Gradients
        error = y_hat - y_train
        dw = (X_train.T @ error) / len(X_train)
        db = np.mean(error)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

    def predict(X):
        probabilities = sigmoid(X @ w + b)
        return (probabilities >= 0.5).astype(int)

    return predict

