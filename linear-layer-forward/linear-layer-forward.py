
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = np.array(X)
    W = np.array(W)
    b = np.array(b)
    xw = X @ W
    y = xw + b
    y = y.tolist()
    return y