from sympy import *
import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.misc import derivative

import progressbar
import time

c = 2.998*10**8 #m.s-1
g = 9.81

la1 = 1.0e-6 #wavelength in vacum 
w0 = 1e-3 # waist 1/e^2 intensity radius
k1 = 2.0*np.pi/la1 # wave-vector
wt1 = 2*np.pi*c/la1
zR1 = np.pi*w0**2/la1 # Rayleigh range


z0 = 0. #position of maximum intensity along the z-axis

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) #?
tL = L/vL

#trap freq
wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(4*U0/(mass*la1**2))/(2*np.pi)

#time to run
tstart = 0.0
tend = 1e-3/tL # !!! time is in unit of tL
dt = 0.0001*tend # 5% period in tL unit

#distance to go
D = 0.5
#variation of the frequency of the counter-propagating beam
a = np.array([[-(tend)**5/30,2*tend], [-(tend)**4/4,1]])
b = np.array([D-g*(tend)**2/3,-g*tend/2])
x = np.linalg.solve(a, b)
alpha = x[0]
beta = x[1]

def wt2(t):
    return wt1-(alpha*(2*t-tend)**4/4-(alpha*(tend)**2-g/tend)*(2*t-tend)**2/2+beta)

#Definition of the gaussian beam
#Waist at z

def w1(z):
    return w0*np.sqrt(1.0+(z/zR1)**2)
def w2(z,t):
    wt2t = wt2(t)
    zR2 = w0**2*wt2t/2/c
    return w0*np.sqrt(1.0+(z/zR2)**2)
    
#Laguerre polynomials

l = 0 #azimuthal index >-1
p = 0 #radial index (positive)

def E1(t,z,y,x):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w1(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR1)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR1))

def E2(t,z,y,x):
    wt2t = wt2(t)
    zR2 = w0**2*wt2t/2/c
    r_w_z_2 = (y**2+x**2)/w2(z,t)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w2(z,t)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR2)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR2))
    
def E_pol(t,z,y,x):
    c = 2.998*10**8 #m.s-1
    wt2t = wt2(t)
    k2 = wt2t/c
    beam1 = E1(t,z,y,x)*np.exp(-1j*k1*z)
    beam2 = E2(t,z,y,x)*np.exp(-1j*k2*z)
    return beam1*np.exp(1j*wt1*t) + np.conjugate(beam2)*np.exp(1j*wt2t*t)

def It(t,z,y,x):
    
    A = E_pol(t,z,y,x)
    return np.vdot(A,A).real

def derivTx(t, z, y, x):
    
    def f(x):
        return It(t, z, y, x)
    return derivative(f, x, dx=1e-11)
    
def derivTy(t, z,y,x):

    def f(y):
        return It(t, z, y, x)
        
    return derivative(f, y, dx=1e-11)
    
def derivTz(t, z,y,x):

    def f(z):
        return It(t, z, y, x)

    return derivative(f, z, dx=10**(-13))

def af2(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-g+derivTz(t*tL, pz*L, py*L, px*L)*L,vz,derivTy(t*tL, pz*L, py*L, px*L)*L, vy, derivTx(t*tL, pz*L, py*L, px*L)*L, vx]

start = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo:")

nb = 20 # nb particles
#vr = maxwell.rvs(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb) #for T = 0.1 milliKelvin
vzr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
vyr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
vxr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
# scale a = sqrt(kT/m) => and v in units of vL = sqrt(2U0/mass) => vL^2/a^2 = 2U0/kT => a = VL sqrt(kT/2U0)
# but I have chosen before U0 = kb* 1mK so a = sqrt(T/(2)) with T in mK

sol_arr=[]

pbar = progressbar.ProgressBar(maxval=len(vzr)).start()

for i,vi in enumerate(vzr):
    vzi = vi
    vyi = vyr[i]
    vxi = vxr[i]
    pbar.update(i)
    
    zi = np.random.normal(0.0,w0/10.0)/L
    yi = np.random.normal(0.0,w0/10.0)/L
    xi = np.random.normal(0.0,w0/10.0)/L

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af2, (tstart,tend), (vzi,zi,vyi,yi,vxi,xi), t_eval = np.linspace(tstart,tend,int((tend-tstart)/dt),endpoint=True),method=BESTMETHOD,atol=1e-5,rtol=1e-6)#,atol=1e-6,rtol=1e-8)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr.append(sol)
    
end = time.time()
pbar.finish()
print('finished in (seconds):', end - start)

print('--- Radau (DP5) ----')
print('message:', sol.message)
print('sol status:', sol.status)
if sol.status ==0:
    print('SUCCESS, reached end.')
print('sol arry', len(sol_arr))
print('sol shape:',sol.y.shape)

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_2gaussian.npz',sol_arr=sol_arr)
print('--done--')
