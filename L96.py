import numpy as np

def L96(x, t):
    # Evaluates the right hand side of the Lorenz '96 system
    F = 8
    
    dx = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F

    return dx