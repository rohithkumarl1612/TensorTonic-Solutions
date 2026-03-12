import numpy as np
import sympy as sp
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    #X(k+1) = x(k) - lr * delta(y)
    x = sp.Symbol('x')
    y = a * (x**2) + b * x + c
    x_new = float(x0)
    for step in range(steps):
        df = sp.diff(y, x)
        result = float(df.subs(x, x_new))
        x_new = x_new - lr * result

    return x_new