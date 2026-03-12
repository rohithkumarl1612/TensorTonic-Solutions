import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X,dtype=float)
    y = np.asarray(y,dtype=float)
    n_features , n_samples = X.shape
    w = np.zeros(n_samples)
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        y_pred = _sigmoid(z)
        dw = (1 / n_samples) * (X.T @ (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        w = w - lr*dw
        b = b - lr*db
    
    return w,b