import numpy as np

def L63(x, t):
    # Evaluates the right hand side of the Lorenz '93 system
    sigma = 10
    rho = 28
    beta = 8/3
    xdot = [sigma*x[1] - sigma*x[0],
            x[0]*rho - x[0]*x[2] - x[1],
            x[0]*x[1] - beta*x[2]]
    return xdot