import numpy as np
np.random.seed()

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar

la = 1.0e-5 #wavelength in vacum 
w0 = 5e-6 # waist 1/e^2 intensity radius
k = 2.0*np.pi/la # wave-vector
zR = np.pi*w0**2/la # Rayleigh range
z0 = 0.0 #position of maximum intensity along the z-axis
g = 9.81
# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) 
tL = L/vL

#trap freq
wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(2*U0/(mass*zR**2))/(2*np.pi)

#time step
dt = 0.05*(1.0/wx)/tL # 5% period in tL unit

#Simulation to analyse :
simulation_name  = "doughnut"

#load the data
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']
nb_part = sol_arr.shape
nb_iter = len(sol_arr[0].t)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

#Analysis plot

time = [t*dt for t in range(nb_iter)]
fig1 = plt.figure(1)

for i in range(nb_part[0]):
    
    simulation_i = sol_arr[i]
    (min_vx, min_vy, min_vz) = (0,0,0)
    
    ax1 = plt.subplot(212)
    ax1.plot(time, simulation_i.y[0,:]*vL)
    ax1.set_title('vz')
    
    if min(simulation_i.y[0,:])<min_vx:
        min_vx = min(simulation_i.y[0,:])

    ax2 = plt.subplot(221)
    ax2.plot(time, simulation_i.y[2,:]*vL)
    ax2.set_title('vy')
    
    if min(simulation_i.y[2,:])<min_vx:
        min_vx = min(simulation_i.y[2,:])

    ax3 = plt.subplot(222)
    ax3.plot(time, simulation_i.y[4,:]*vL)
    ax3.set_title('vx')
    
    if min(simulation_i.y[4,:])<min_vx:
        min_vx = min(simulation_i.y[4,:])

fig2 = plt.figure(2)
target = (-0.2, 1e-3)
target_reached = 0
def findstep(list, value):
    n = len(list)
    for i in range(n):
        if list[i]<value:
            return i
    return False

for i in range(nb_part[0]):
    
    simulation_i = sol_arr[i]
    
    ax4 = plt.subplot(312)
    ax4.plot(time, simulation_i.y[1,:]*L)
    ax4.set_title('z')
    (min_x,min_y,min_z) = (0,0,0)
    
    if min(simulation_i.y[1,:])<min_vx:
        min_x = min(simulation_i.y[1,:])

    ax5 = plt.subplot(321)
    ax5.plot(time, simulation_i.y[3,:]*L)
    ax5.set_title('y')

    if min(simulation_i.y[3,:])<min_vx:
        min_y = min(simulation_i.y[3,:])

    ax6 = plt.subplot(322)
    ax6.plot(time, simulation_i.y[5,:]*L)
    ax6.set_title('x')

    if min(simulation_i.y[5,:])<min_vx:
        min_z = min(simulation_i.y[5,:])
        
    if min(simulation_i.y[1,:])*L<target[0]:
        
        index = findstep(simulation_i.y[1,:], target[0])
        
        if (simulation_i.y[3,:][index]*L)**2+(simulation_i.y[5,:][index]*L)**2<target[1]**2:
            target_reached += 1

print(min_x*L, min_y*L, min_z*L, min_vx*L, min_vy*L, min_vz*L)
print(target_reached)
plt.show()

