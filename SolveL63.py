import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def L63(x, t):
    sigma = 10
    rho = 28
    beta = 8/3
    xdot = [sigma*x[1] - sigma*x[0],
            x[0]*rho - x[0]*x[2] - x[1],
            x[0]*x[1] - beta*x[2]]
    return xdot

Tspan = np.array([0, 50])
v0 = np.array([1, 0, 0])
tout = np.linspace(Tspan[0], Tspan[1], 1000)
vout = odeint(L63, v0, tout)

# Plot in 3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(vout[:,0], vout[:,1], vout[:,2], 'b')
v0 = np.array([1+1E-5, 0, 0])
vout = odeint(L63, v0, tout)
ax.plot(vout[:,0], vout[:,1], vout[:,2], 'r')

# Plot of (x,y,z) vs time
fig = plt.figure(2)
plt.plot(tout, vout[:,0], tout, vout[:,1], tout, vout[:,2])
v0 = np.array([1, 0, 0])
vout = odeint(L63, v0, tout)
plt.plot(tout, vout[:,0], tout, vout[:,1], tout, vout[:,2])
plt.legend(['xp', 'yp', 'zp', 'x', 'y', 'z'])
plt.show()

