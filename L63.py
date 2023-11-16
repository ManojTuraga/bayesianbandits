import numpy as np

def L63(x, t):
    xdot = np.array([
        10 * x[1] - 10 * x[0],
        x[0] * 28 - x[0] * x[2] - x[1],
        x[0] * x[1] - 8/3 * x[2]
    ])
    return xdot