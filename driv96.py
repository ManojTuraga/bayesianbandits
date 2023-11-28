import numpy as np
from scipy.integrate import odeint
from L96 import L96
import shutup
shutup.please()

#Initialization
n=40 #Dimension of state space
m=20 #Dimension of observation space
N=40 #Number of ensemble members
H = np.zeros((m,n))
for k in range(m):
    for j in range(n):
        if not j % 2:
            H[k][j]=1

dt = 0.1 #Time between observations
J = 1000 #Number of assimilation times
vt = np.zeros((n,J+1))
yt = np.zeros((m,J+1))
vt[0,0]=1
Varr = np.zeros((n,N))
Vharr = np.zeros((n,N))

#Covariances
alpha=0.1
beta =0.1
C0 = beta**2*np.eye(n)
Sigma = beta**2*np.eye(n)
Gamma = alpha**2*np.eye(m)

RMSE = []

# Define the Gaspari-Cohn function
def gc_dist(dist, r):
    if dist <= 0:
        return 1
    elif dist > r:
        return 0
    else:
        return 1/4 * (2 - (dist/r)**2 + (dist/r)**3 + 1/2 * (dist/r)**4 - 1/2 * (dist/r)**5) if dist <= r/2 else 1/2 * (1 + (1 - dist/r)**3 - (1 - dist/r)**4 - 1/2 * (1 - dist/r)**5)

# Define the localization radius
r = 5

# Compute the localization matrix
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist = abs(i - j)
        C[i, j] = gc_dist(dist, r)

#Get truth and synthetic observations
for j in range(J):
    Tspan = [j*dt,(j+1)*dt]
    w = odeint(L96,vt[:,j],Tspan)
    vt[:,j+1]=w[-1,:]
    yt[:,j+1]=H@vt[:,j+1]+alpha*np.random.randn(m)

#ICs for ensembles
for k in range(N):
    Varr[:,k] = vt[:,0] + beta*np.random.randn(n)

#Main Time Loop
for j in range(J):

    #Prediction of Ensembles
    for k in range(N):
        Tspan = [j*dt,(j+1)*dt]
        w = odeint(L96,Varr[:,k],Tspan)
        Vharr[:,k]=w[-1,:] + beta*np.random.randn(n) #* np.sqrt(1 + alpha**2 / beta**2)

    #Sample mean
    mhat = np.zeros(n)
    for k in range(N):
        mhat = mhat + Vharr[:,k]
    mhat = mhat/N

    #Sample covariance
    Chat = np.zeros((n,n))
    for k in range(N):
        covvec = Vharr[:,k]-mhat
        Chat = Chat + covvec[:,None]@covvec[None,:]
    Chat = Chat/(N-1)

    # Apply the localization
    Chat = C * Chat

    #Analysis
    S = H@Chat@H.T+Gamma
    Innov = []
    for k in range(N):
        temp = (yt[:,j+1]-H@Vharr[:,k]+alpha*np.random.randn(1,m))[0]
        Innov.append( temp )
    
    Innov = np.array( Innov )

    SinvI = np.linalg.solve(S,Innov.T)

    KI = Chat@H.T@SinvI
    Varr = Vharr+KI.T
    #No Analysis: comment the line below and uncomment the line above to add the Analysis
    #Varr = Vharr;

    #Sample mean of the Analysis!
    mVarr = np.zeros(n)
    for k in range(N):
        mVarr = mVarr + Varr[:,k]
    mVarr = mVarr/N

    #Record RMSE
    RMSE.append( np.linalg.norm(mVarr - vt[:,j+1])/np.sqrt(n))

#Plot RMSE
import matplotlib.pyplot as plt
plt.plot(np.arange(J),RMSE)
plt.plot(np.arange(J),alpha*np.ones(J),'g-')
plt.title('RMSE')
plt.savefig( 'rmse.png' )

RMSE = np.mean(RMSE)
print(f"RMSE: {RMSE}")