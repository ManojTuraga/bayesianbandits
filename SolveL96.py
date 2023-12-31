import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from L96 import L96

Tspan = np.array([0, 50])
v0 = np.array([1] + [0]*39)
tout = np.linspace(Tspan[0], Tspan[1], 1000)
vout = odeint(L96, v0, tout)

# Plot in 3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot(vout[:,0], vout[:,1], vout[:,2], 'b')
v0 = np.array([1+1E-5, 0, 0])
vout = odeint(L96, v0, tout)
ax.plot(vout[:,0], vout[:,1], vout[:,2], 'r')
ax.set_title("3D plot of Lorenz 96 model")

# Plot of (x,y,z) vs time
fig = plt.figure(2)
plt.plot(tout, vout[:,0], tout, vout[:,1], tout, vout[:,2])
v0 = np.array([1, 0, 0])
vout = odeint(L96, v0, tout)
plt.plot(tout, vout[:,0], tout, vout[:,1], tout, vout[:,2])
plt.title("Plot of (x,y,z) vs time")
plt.legend(['xp', 'yp', 'zp', 'x', 'y', 'z'])
plt.show()
