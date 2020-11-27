import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar
import time

#simulation to run
simulation_name = "doughnut_0.3_100_10"

#Fundamental Physical Constants
c = 2.99792458e8 #m.s-1
mu0 = 4*np.pi*1e-7 #N/A²
hbar = 1.054571628e-34 #J.s
kb = 1.3806504e-23 # J/K # Boltzmann constant
g = 9.81 #m.s-2

#Cesium Physical Properties
mass = 2.20694657e-25 #kg #cesium mass

#Cesium D2 Transition Optical Properties
wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

#Beam Properties
delta = 1e10 #Hz
wl = wa + delta #Hz
k = wl/c #m-1
la = 2.0*np.pi/k #m # wave-vector
w0 = 0.3e-3 #m # waist 1/e^2 intensity radius
zR = np.pi*w0**2/la #m # Rayleigh range
z0 = 0.0 #m #position of maximum intensity along the z-axis

print("la, wl, delta ", la, wl, delta)

#Doughnut beam : described using Laguerre polynomials of parameter l=1, p=0. 
l = 1 #azimuthal index >-1
p = 0 #radial index (positive)

#Potential depth in Joules, U0 = kB*T
P0 = 100e-3 #W #Laser power
I0 = 2*P0/np.pi/w0**2 #W/m**2 #Laser intensity
U0 = 3*np.pi*c**2/2/wa**3*gamma/delta*I0 #J #Potential depth
Tp = U0/kb #°K #Trapping temperature 
print("Tp", Tp)

# Length, speed, time constants
vL = np.sqrt(2*U0/mass)
tL = np.sqrt(mass/U0)*w0/2
L = w0/np.sqrt(2)
print("L, vL, tL", L, vL, tL)

#Cesium cloud parameters
nb = 1000 # nb particles
Ti = 1e-4 #K #Initial temperature of the cloud #Doppler Temperature

#Simulation's time duration and step
#Definition of the simulation's duration
D = 0.5 #distance to achieve
coeff = [-g/2, np.sqrt(kb*Ti/mass), 0.5]
possible_tend = np.roots(coeff)
if possible_tend[0]<0:
    tend = possible_tend[1]/tL
else :
    tend = possible_tend[0]/tL

#Simulation's starting time
tstart = 0.0
#Simulation's time step
dt = (0.05)**2*tend # 5% period in tL unit

print("tend, dt : ", tend*tL, dt*tL)

#initial speeds in vL units #Maxwell-Botzmann distribution
vzr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
vyr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
vxr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
# v/vL = sqrt(kbTi/m)/sqrt(2kbTp/m) = sqrt(Ti/(2Tp))

#Definition of the gaussian beam

#Waist at z
def w(z):
    return w0*np.sqrt(1.0+((z-z0)/zR)**2)

#Intensity of the beam    
def It(z,y,x):
    #Unitary intensity
    #Amplitude added in the force expression
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2
    
def derivTx(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return (w0/w(z))**2*(2*r_w_z_2)**(l-1)*np.exp(-2*r_w_z_2)*lag*(4*x/w(z)**2)*((l-2*r_w_z_2)*lag+4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))
    
def derivTy(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return (w0/w(z))**2*(2*r_w_z_2)**(l-1)*np.exp(-2*r_w_z_2)*lag*(4*y/w(z)**2)*((l-2*r_w_z_2)*lag+4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))
    
def derivTz(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return np.exp(-2*r_w_z_2)*lag*2*(2*r_w_z_2)**l*la**2*z/np.pi**2/w(z)**4*((-(1+l)+2*r_w_z_2)*lag-4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))


def af2(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-g*tL/vL-U0/mass*tL/vL*derivTz(pz*L,py*L,px*L), vz, -U0/mass*tL/vL*derivTy(pz*L,py*L,px*L), vy, -U0/mass*tL/vL*derivTx(pz*L,py*L,px*L), vx]

start = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo:")

sol_arr=[]

pbar = progressbar.ProgressBar(maxval=len(vzr)).start()

for i,vi in enumerate(vzr):
    vzi = vi
    vyi = vyr[i]
    vxi = vxr[i]
    pbar.update(i)
    
    zi = np.random.uniform(-0.5e-3,0.5e-3)/L
    yi = np.random.uniform(-0.5e-3,0.5e-3)/L
    xi = np.random.uniform(-0.5e-3,0.5e-3)/L

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
iter = len(sol_arr)
print('sol array', iter)
print('sol shape:',sol.y.shape)

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_'+simulation_name+'.npz',sol_arr=sol_arr)
print('--done--')