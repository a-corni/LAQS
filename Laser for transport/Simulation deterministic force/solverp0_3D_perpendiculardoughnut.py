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

w0 = 5e-6 # waist 1/e^2 intensity radius
z0 = 0. #position of maximum intensity along the z-axis

la1 = 1.0e-6 #wavelength in vacum 
k1 = 2.0*np.pi/la1 # wave-vector
w1 = 2*np.pi*c/la1
zR1 = np.pi*w0**2/la1 # Rayleigh range

la2 = 1.5e-6 #wavelength in vacum 
k2 = 2.0*np.pi/la2 # wave-vector
w2 = 2*np.pi*c/la2
zR2 = np.pi*w0**2/la2 # Rayleigh range

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
wz = np.sqrt(2*U0/(mass*zR1**2))/(2*np.pi)

print(wx,wz)

print("L,vL,tL",L,vL,tL)

#Definition of the gaussian beam
#Waist at z

def w1(z):
    return w0*np.sqrt(1.0+(z/zR1)**2)

def w2(z):
    return w0*np.sqrt(1.0+(z/zR2)**2)
    
#Laguerre polynomials

l = 1 #azimuthal index >-1
p = 0 #radial index (positive)

def E1(z,y,x):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w1(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR1)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR1))

def E2(z,y,x):
    r_w_z_2 = (y**2+x**2)/w2(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w2(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR2)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR2))
    
def E_pol(z,y,x):
    pol1 = np.array([0,1j/np.sqrt(2),1/np.sqrt(2)])
    beam1 = E1(z,y,x)*np.exp(-1j*k1*z)*pol1
    pol2 = np.array([1j/np.sqrt(2),1/np.sqrt(2),0])
    beam2 = E2(x,z,y)*np.exp(-1j*k2*x)
    return beam1 + np.conjugate(beam2)*pol2

def It(z,y,x):
    A = E_pol(z,y,x)
    return np.vdot(A,A)

def I_gaussian(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2

def derivTx(z,y,x):
    
    def f(x):
        return It(z,y,x)
    return derivative(f, x, dx=1e-11)
    
def derivTy(z,y,x):

    def f(y):
        return It(z,y,x)
        
    return derivative(f, y, dx=1e-11)
    
def derivTz(z,y,x):

    def f(z):
        return It(z,y,x)

    return derivative(f, z, dx=10**(-10))

def af2(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-derivTz(pz*L,py*L,px*L)*L,vz,-derivTy(pz*L,py*L,px*L)*L,vy,-derivTx(pz*L,py*L,px*L)*L,vx]


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

tstart = 0.0
nper = 20 # nb oscill periods
tend = nper*(1.0/wx)/tL # !!! time is in unit of tL
dt = 0.05*(1.0/wx)/tL # 5% period in tL unit

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
    
    sol = solve_ivp(af2, (tstart,tend), (vzi,zi,vyi,yi,vxi,xi), t_eval = np.linspace(tstart,tend,int((tend-tstart)/dt),endpoint=True),method=BESTMETHOD,atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr.append(sol)
    
end = time.time()
pbar.finish()
print('finished in (seconds):', end - start)

print('--- rk45 (DP5) ----')
print('message:', sol.message)
print('sol status:', sol.status)
if sol.status ==0:
    print('SUCCESS, reached end.')
print('sol arry', len(sol_arr))
print('sol shape:',sol.y.shape)

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_perpendiculardoughnut.npz',sol_arr=sol_arr)
print('--done--')
