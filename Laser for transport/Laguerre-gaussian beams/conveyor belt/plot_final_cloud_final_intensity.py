from sympy import *
import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.misc import derivative

import progressbar
from matplotlib import pyplot as plt

#Definition of the gaussian beam
g = 9.81
c = 2.998*10**8 #m.s-1
wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

#Beam Properties
delta1 = -1e9 #Hz
wl1 = wa + delta1 #Hz
k1 = wl1/c #m-1
la1 = 2.0*np.pi/k1 #m # wave-vector
print(la1/2)
w0 = 1e-3 #m # waist 1/e^2 intensity radius
zR1 = np.pi*w0**2/la1 #m # Rayleigh range
print(zR1)
z0 = 0.

z0 = 0. #position of maximum intensity along the z-axis

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) #?
tL = L/vL


wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(2*U0/(mass*zR1**2))/(2*np.pi)
tstart = 0.0
nper = 1 #20 # nb oscill periods
tend_sim = nper*(1.0/wx)/tL # !!! time is in unit of tL
tend = tend_sim*tL
dt = 0.05*(1.0/wx)/tL # 5% period in tL unit
time = [k*dt for k in range(int((tend_sim/2-tstart)/dt))]
print(tend)

def v(t):
    D = 0.1
    if t<=tend/4:
        return D/tend*(-7040/36*(t/tend)**4+320/3*(t/tend)**3)
    elif t<=3*tend/4 :
        return D/tend*(3200/36*(t/tend)**4-1600/9*(t/tend)**3+640/6*(t/tend)**2-160/9*(t/tend))-D/tend*(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4))+D/tend*(-7040/36*(1/4)**4+320/3*(1/4)**3)
    else :
        return D/tend*(-7040/36*(t/tend)**4+6080/9*(t/tend)**3-5120/6*(t/tend)**2+4160/9*t/tend)-D/tend*(-7040/36+6080/9-5120/6+4160/9)

def pos(t):
    D = 0.1
    if t<=tend/4:
        return D*(-7040/180*(t/tend)**5+320/12*(t/tend)**4)
    elif t<=3*tend/4 :
        return D*(3200/180*(t/tend)**5-1600/36*(t/tend)**4+640/18*(t/tend)**3-160/18*(t/tend)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*t/tend)+D*(-7040/180*(1/4)**5+320/12*(1/4)**4)-D*(3200/180*(1/4)**5-1600/36*(1/4)**4+640/18*(1/4)**3-160/18*(1/4)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*1/4)
    else :
        return D*(-7040/180*(t/tend)**5+6080/36*(t/tend)**4-5120/18*(t/tend)**3+4160/18*(t/tend)**2-(-7040/36+6080/9-5120/6+4160/9)*t/tend)-D*(-7040/180+6080/36-5120/18+4160/18-(-7040/36+6080/9-5120/6+4160/9))+D
 
def dw(t):
    return -2*np.pi*2/la1*v(t)

def w1(z):
    return w0*np.sqrt(1.0+((z-z0)/zR1)**2)
#Definition of the gaussian beam
#Waist at z

#Laguerre polynomials

l = 0 #azimuthal index >-1
p = 0 #radial index (positive)

def E(z,y,x):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return w0/w1(z)*np.exp(-r_w_z_2)*np.exp(-1j*k1*z-1j*r_w_z_2*z*w0**2/2/zR1**2)*np.exp(1j*np.arctan(z/zR1))
    
def E_pol(z,y,x,t):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return w0/w1(z)*np.exp(-r_w_z_2)*(1+np.exp(-2j*k1*z-2j*r_w_z_2*z*w0**2/2/zR1**2)*np.exp(2j*np.arctan(z/zR1))*np.exp(1j*dw(t)))

def It(z,y,x,t):
    return np.abs(E_pol(z,y,x,t))**2

simulation_name = "2gaussian"
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']
nb_part = sol_arr.shape[0]
nb_iter = len(sol_arr[0].t)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter)


for k in range(0,nb_iter):
    plt.figure(k)
    Z = [sol_arr[0].y[1,k]*L+10*la1/2*(i-5000)/10000 for i in range(10000)]
    plt.plot(Z,[It(z,0,0,time[k]) for z in Z]) 
    
    pz = [sol_arr[j].y[1,k]*L for j in range(nb_part)]
    py = [sol_arr[j].y[3,k]*L for j in range(nb_part)]
    plt.scatter(pz,py, s = 0.5, c = 'r')
    
    plt.title("Position and intensity of the beam at time "+'%.5f'%(time[k])+" s")
    plt.xlabel("z axis (m)")
    plt.ylabel("Intensity along z, at time t = "+"%.5f"%(time[-1])+" s")
    plt.savefig("D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/2gaussian/intensity_it_"+str(k))

plt.close()
