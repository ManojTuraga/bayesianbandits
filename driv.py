import numpy as np
from scipy.integrate import odeint
from L63 import L63
import matplotlib.pyplot as plt

# equivalent to the driv96.m file

# Initialization
n = 3  # Dimension of state space
m = 1  # Dimension of observation space
N = 10  # Number of ensemble members
H = np.zeros((m, n)) # Opservation matrix
H[0, 0] = 1  # Observe x
H[0, 1] = 1  # Observe y
H[0, 2] = 1  # Observe z

dt = 0.1  # Time between observations
J = 10000  # Number of assimilation times
vt = np.zeros((n, J + 1))
yt = np.zeros((m, J + 1))
vt[:, 0] = 1

Varr = np.zeros((n, N))
Vharr = np.zeros((n, N))

# Covariances
alpha = 1
beta = 0.1
C0 = beta**2 * np.eye(n)
Sigma = beta**2 * np.eye(n)
Gamma = alpha**2 * np.eye(m)

# Get truth and synthetic observations
for j in range(J):
    Tspan = [j * dt, (j + 1) * dt]
    w = odeint(L63, vt[:, j], Tspan)[-1, :]
    vt[:, j + 1] = w
    yt[:, j + 1] = np.dot(H, vt[:, j + 1]) + alpha * np.random.randn(m)

# ICs for ensembles
for k in range(N):
    Varr[:, k] = vt[:, 0] + beta * np.random.randn(n)

# Main Time Loop
RMSE = np.zeros(J)
for j in range(J):
    # Prediction of Ensembles
    for k in range(N):
        Tspan = [j * dt, (j + 1) * dt]
        w = odeint(L63, Varr[:, k], Tspan)[-1, :]
        Vharr[:, k] = w + beta * np.random.randn(n)

    # Sample mean
    mhat = np.mean(Vharr, axis=1)

    # Sample covariance
    Chat = np.cov(Vharr)

    # Analysis
    S = np.dot(np.dot(H, Chat), H.T) + Gamma
    print( yt[:, j + 1] )
    print( np.dot(H, Vharr) + alpha * np.random.randn(N) )
    Innov = yt[:, j + 1] - np.dot(H, Vharr) + alpha * np.random.randn(N)
    SinvI = np.linalg.solve(S, Innov)

    # Kalman Gain
    KI = np.dot(np.dot(Chat, H.T), SinvI)

    # Update Ensembles
    Varr = Vharr + KI

    # Sample mean of the Analysis
    mVarr = np.mean(Varr, axis=1)

    # Calculate RMSE
    RMSE[j] = np.linalg.norm(mVarr - vt[:, j + 1]) / np.sqrt(n)


# Plot RMSE
plt.figure()
plt.plot(range(1, J + 1), RMSE)
plt.axhline(alpha, color='g', linestyle='-')
plt.title('RMSE')
plt.savefig( 'rmse.png' )

meanVarr = Vharr
RMSE = np.mean(RMSE)
print(f"RMSE: {RMSE}")