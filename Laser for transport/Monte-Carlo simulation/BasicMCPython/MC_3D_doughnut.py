import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar
import time

#simulation to run
simulation_name = "doughnut_1_100_10"

import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar
import time

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
print("Maximum scattering rate : ", gamma_scatter_max)

# Length, speed, time constants
vL = 1 #np.sqrt(2*U0/mass)
tL = 1 #np.sqrt(mass/U0)*w0/2
L = 1 #w0/np.sqrt(2)
print("L, vL, tL", L, vL, tL)

#Cesium cloud parameters
nb_part = 1000 # nb particles
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
nb_samples = 400 #algorithm will sample nb_samples+1 times
dt_max = 1/(nb_samples)*tend2 # 0.25% of simulation time
dt_min = 1/gamma_scatter_max/tL
print("tend1 (in s) :", tend1*tL)
print("tend2, dt_max (in s), dt_min(in s): ", tend2*tL, dt_max*tL, dt_min*tL)
print("Maximum number of scattering for 1 atom during the simulation ", gamma_scatter_max*tend2*tL/2/np.pi)
print("Maximum number of scattering for 1 atom between two samples ", gamma_scatter_max*dt_max*tL/2/np.pi)


#initial speeds in vL units #Maxwell-Botzmann distribution
vzr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb_part)/vL
vyr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb_part)/vL
vxr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb_part)/vL
thetar = np.random.uniform(0, np.pi, size = nb_part)
phir = np.random.uniform(0, 2*np.pi, size = nb_part)
rr = np.random.uniform(0,1e-3)/L
zr = rr*np.cos(thetar)
yr = rr*np.sin(thetar)*np.sin(phir)
xr = rr*np.sin(thetar)*np.cos(phir)
# v/vL = sqrt(kbTi/m)/sqrt(2kbTp/m) = sqrt(Ti/(2Tp))

#Definition of the gaussian beam
#Waist at z
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


start = time.time()
mean_duration = 0
sol = np.empty([6*nb_part,nb_samples+1])
sol_t = np.array([(vzr[i], zr[i], vyr[i], yr[i], vxr[i], xr[i]) for i in range(nb_part)] )
nb_wrong_proba = 0
print(np.shape(sol_t))

for i in range(nb_part):
    
    #initialize
    start_i = time.time()
    t_i = tstart1 #transport time
    dt = min(1/gamma_scatter_max/tL, dt_max) #simulation time step
    pbar2 = progressbar.ProgressBar(maxval=nb_samples).start() #progress bar

    #First sample : t = 0 
    sample = 0
    sol[6*i:6*i+6,sample] = sol_t[i] 
    
    while sample < nb_samples: 
        
        t_i += dt
        pbar2.update(sample)
        
        (vz, pz, vy, py, vx, px) = sol_t[i]
        pz = pz + vz*dt
        py = py + vy*dt
        px = px + vx*dt
        
        scattering_event = np.random.uniform(0,1)
        scattering_rate =  gamma_scatter_0*It(pz*L,py*L,px*L)/2/np.pi #in Hz
        proba_scatter = scattering_rate*dt*tL
        
        if proba_scatter > 1 or proba_scatter < 0:
            print("Abnormal process, used proba : ", proba_scatter)
            nb_wrong_proba +=1
            break
        
        # if scattering_event<proba_scatter :
        #     
        #     thetai = np.random.uniform(0, np.pi)
        #     phii = np.random.uniform(0, 2*np.pi)
        #     vz = vz + hbar*k/mass*(1+np.cos(thetai))/vL
        #     vy = vy + hbar*k/mass*np.sin(thetai)*np.sin(phii)/vL
        #     vx = vx + hbar*k/mass*np.sin(thetai)*np.cos(phii)/vL
        
        vz = vz - g*dt*tL/vL - U0/mass*dt*tL/vL*derivTz(pz*L,py*L,px*L)
        vy = vy - U0/mass*dt*tL/vL*derivTy(pz*L,py*L,px*L)
        vx = vx - U0/mass*dt*tL/vL*derivTx(pz*L,py*L,px*L)
        
        sol_t[i] = [vz, pz, vy, py, vx, px]
        
        gradI_v = derivTx(pz*L, py*L, px*L)*vx + derivTy(pz*L, py*L, px*L)*vy + derivTz(pz*L, py*L, px*L)*vz
        
        #methode 1
        #dt = min(0.1/scattering_rate/tL, dt_max)
        #methode 2
        dt = min(dt_max, 0.1/(scattering_rate+gamma_scatter_0*gradI_v*dt*L)/tL)
        dt = max(dt, dt_min)
        #methode 3
        #if gradI_v > 0 : 
        #   dt = min(scattering_rate/gamma_scatter_0*np.pi/gradI_v/L*(-1+np.sqrt(1+2*0.1*gradI_v*vL/(scattering_rate/gamma_scatter_0*2*np.pi)**2), dt_max)
        #   dt = max(dt, dt_min)
        #elif gradI_v == 0 : 
        #    dt = min(0.1/scattering_rate/tL, dt_max)
        #    dt = max(dt, dt_min) 
        #else :
        #    dt = min(dt_max, scattering_rate/gamma_scatter_0*np.pi/gradI_v/L)
        #    dt = max(dt, dt_min)
        
        if t_i >= (sample+1)*dt_max :
            sample+=1
            sol[6*i:6*i+6,sample] = [vz, pz, vy, py, vx, px]
    
    pbar2.finish()
    end_i = time.time()
    duration_i = end_i - start_i
    mean_duration = (mean_duration*i + duration_i)/(i+1)
    print("Simulation ", i,"finished, mean duration time : ", mean_duration)    
            
end = time.time()
print("Number of abnormal process : ", nb_wrong_proba)
print("Mean duration time : ", mean_duration)
print("Simulation ended in :", end-start)
print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/MC_3D_'+simulation_name+'.npz',sol_arr=sol)
print('--done--')

    
