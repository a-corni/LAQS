import numpy as np
np.random.seed()

from MC_ivp_2 import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar
import time

#simulation to run
simulation_name = "doughnut_1_100_10"

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
delta = 10e9 #Hz
wl = wa + delta #Hz
k = wl/c #m-1
la = 2.0*np.pi/k #m # wave-vector
w0 = 1e-3 #m # waist 1/e^2 intensity radius
zR = np.pi*w0**2/la #m # Rayleigh range
z0 = 0.0 #m #position of maximum intensity along the z-axis

print("la, wl, delta ", la, wl, delta)

#Doughnut beam : described using Laguerre polynomials of parameter l=1, p=0. 
l = 1 #azimuthal index >-1
p = 0 #radial index (positive)

# Potential depth in Joules, U0 = kB*T
P0 = 100e-3 #W #Laser power
I0 = 2*P0/np.pi/w0**2 #W/m**2 #Laser intensity
Imax = I0*np.exp(-1)
U0 = 3*np.pi*c**2/2/wa**3*gamma/delta*I0
Umax = 3*np.pi*c**2/2/wa**3*gamma/delta*Imax #J #Potential depth
Tp = Umax/kb #°K #Trapping potential in Temperature 
print("Tp", Tp)

# Scattering rate
gamma_scatter_0 = U0*gamma/delta/hbar
gamma_scatter_max = Umax*gamma/delta/hbar
print("Maximum scattering rate (rad.s-1): ", gamma_scatter_max)

# Length, speed, time constants
vL = 1     #= np.sqrt(2*U0/mass)
tL = 1     #= np.sqrt(mass/U0)*w0/2
L = 1      #= w0/np.sqrt(2)
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
    tend2 = possible_tend[1]/tL
    
else :    
    tend2 = possible_tend[0]/tL
    
#First part of the simulation : dispersion of the cloud

tstart1 = 0.0
tend1 = 0.01*tend2 #1% of simulation time
tstart2 = tend1

#Simulation's time step

dt = 0.0025*tend2 # 0.25% of simulation time
print("tend1 (in s) :", tend1*tL)
print("tend2, dt (in s): ", tend2*tL, dt*tL)
print("Maximum number of scattering for 1 atom during the simulation", gamma_scatter_max*tend2*tL/2/np.pi)
print("Maximum number of scattering for 1 atom during one step", gamma_scatter_max*dt*tL/2/np.pi)

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

def fun_scattering(t,y):
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    mu = 2*np.random.uniform(0,1)-1 #cos(theta)
    phi = 2*np.random.uniform(0,1)*np.pi
    vz += hbar*k/mass*(1 + np.sqrt(1-mu**2)*np.cos(phi))
    vy += hbar*k/mass*np.sqrt(1-mu**2)*np.sin(phi)
    vx += hbar*k/mass*mu
    return [vz, pz, vy, py, vx, px]

def proba_scattering(dt, y):
    
    return gamma_scatter_0*It(y[1], y[3], y[5])*dt
    
def af1(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-g*tL/vL-U0/mass*tL/vL*derivTz(pz*L,py*L,px*L), vz, -U0/mass*tL/vL*derivTy(pz*L,py*L,px*L), vy, -U0/mass*tL/vL*derivTx(pz*L,py*L,px*L), vx]

start1 = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo 1:")

sol_arr_1 = []

pbar = progressbar.ProgressBar(maxval=len(vzr)).start()

for i,vi in enumerate(vzr):
    vzi = vi
    vyi = vyr[i]
    vxi = vxr[i]
    pbar.update(i)
    
    thetai = np.random.uniform(0, np.pi)
    phii = np.random.uniform(0, 2*np.pi)
    ri = np.random.uniform(0,1e-3)/L
    zi = ri*np.cos(thetai)
    yi = ri*np.sin(thetai)*np.sin(phii)
    xi = ri*np.sin(thetai)*np.cos(phii)

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af1, fun_scattering, proba_scattering, (tstart1,tend1), (vzi,zi,vyi,yi,vxi,xi), method = BESTMETHOD, t_eval = np.linspace(tstart1,tend1,int((tend1-tstart1)/dt),endpoint=True),atol = 1e-10, rtol = 1e-6, dense_output = False, events = None, vectorized = False)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr_1.append(sol)

nb_iter = sol_arr_1[0].y.shape[1]
nb_part = len(sol_arr_1)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

end1 = time.time()
pbar.finish()
print('Part 1 finished in (seconds):', end1 - start1)

print("start Dorman-Prince 5 algo 2:")
zr = []
yr = []
xr = []
vzr = []
vyr = []
vxr  = []

for i in range(nb_part-1):
    zri = sol_arr_1[i].y[1,nb_iter-1] 
    yri = sol_arr_1[i].y[3,nb_iter-1] 
    xri = sol_arr_1[i].y[5,nb_iter-1] 
    vzri = sol_arr_1[i].y[0,nb_iter-1] 
    vyri = sol_arr_1[i].y[2,nb_iter-1]
    vxri = sol_arr_1[i].y[4,nb_iter-1]

    if (yri*L)**2+(xri*L)**2<w0**2/2:
        if 1/2*mass*((vyri*vL)**2+(vxri*vL)**2) < Umax:
            zr.append(zri)
            yr.append(yri)
            xr.append(xri)
            vzr.append(vzri)
            vyr.append(vyri)
            vxr.append(vxri)

def af2(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-g*tL/vL-U0/mass*tL/vL*derivTz(pz*L,py*L,px*L), vz, -U0/mass*tL/vL*derivTy(pz*L,py*L,px*L), vy, -U0/mass*tL/vL*derivTx(pz*L,py*L,px*L), vx]

start2 = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

sol_arr_2 = []

pbar = progressbar.ProgressBar(maxval=len(vzr)).start()
print("Number of articles after filtering :", len(vzr))
for i,vi in enumerate(vzr):
    vzi = vi
    vyi = vyr[i]
    vxi = vxr[i]
    pbar.update(i)
    
    zi = zr[i]
    yi = yr[i]
    xi = xr[i]

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af2, fun_scattering, proba_scattering, (tstart2,tend2), (vzi,zi,vyi,yi,vxi,xi), method=BESTMETHOD, t_eval = np.linspace(tstart2,tend2,int((tend2-tstart2)/dt),endpoint=True), atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr_2.append(sol)
    
end2 = time.time()
pbar.finish()
print('Part 1 finished in (seconds):', end2 - start2)

print('--- rk45 (DP5) ----')
print('message:', sol.message)
print('sol status:', sol.status)
if sol.status ==0:
    print('SUCCESS, reached end.')
    
nb_iter = sol_arr_2[0].y.shape[1]
nb_part = len(sol_arr_2)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_'+simulation_name+'.npz',sol_arr=sol_arr_2)
print('--done--')