from matplotlib import pyplot as plt
from statistics import *
import numpy as np

#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.stats import norm

import time

#simulation to run
simulation_name = "doughnut_1_1_20"
print("Reading results for simulation : ", simulation_name)

w0 = 1e-3
Tp = 1e-3
delta = 20e9
kb = 1.3806504e-23 # J/K # Boltzmann constant
U0 = kb*Tp*np.exp(1) #W.


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
wl = wa + delta #Hz
k = wl/c #m-1
la = 2.0*np.pi/k #m # wave-vector
zR = np.pi*w0**2/la #m # Rayleigh range
z0 = 0.0 #m #position of maximum intensity along the z-axis

print("la, wl, delta ", la, wl, delta)

#Doughnut beam : described using Laguerre polynomials of parameter l=1, p=0. 
l = 1 #azimuthal index >-1
p = 0 #radial index (positive)

# Potential depth in Joules, U0 = kB*T
P0 = np.pi*w0**2/2*U0*delta/gamma*2*wa**3/3/np.pi/c**2
I0 = 2*P0/np.pi/w0**2 #W/m**2 #Laser intensity
Imax = I0*np.exp(-1)
Umax = 3*np.pi*c**2/2/wa**3*gamma/delta*Imax #J #Potential depth
Tp = Umax/kb #°K #Trapping potential in Temperature 
print("Tp", Tp)
print("P0", P0)

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
Ti = 1e-4 #K #Initial temperature of the cloud #Doppler Temperature

#Simulation's time duration and step
#Definition of the simulation's duration
D = 0.5 #distance to achieve
coeff = [-(gamma_scatter_max*hbar*k/mass+g)/2, np.sqrt(kb*Ti/mass), 5.5*D]
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

#load the data
distribution_init_z = []
distribution_init_y = []
distribution_init_x = []
distribution_init_vz = []
distribution_init_vy = []
distribution_init_vx = []

data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MC_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']

nb_part = sol_arr.shape[0]
nb_iter = len(sol_arr[0].t)

#Analysis plot

distribution_init_z = [sol_arr[i].y[1,0]*L for i in range(nb_part)]
distribution_init_y = [sol_arr[i].y[3,0]*L for i in range(nb_part)]
distribution_init_x = [sol_arr[i].y[5,0]*L for i in range(nb_part)]
distribution_init_vz = [sol_arr[i].y[0,0]*vL for i in range(nb_part)]
distribution_init_vy = [sol_arr[i].y[2,0]*vL for i in range(nb_part)]
distribution_init_vx = [sol_arr[i].y[4,0]*vL for i in range(nb_part)]
    
print("Nb part :", nb_part)
print("Nb iteration :", nb_iter) 


#display the initial speed distribution
fig1 = plt.figure(1)

ax1 = plt.subplot(311)
nvzi, binsvzi, patchesvzi = ax1.hist(distribution_init_vz, density = True)
muvzi = mean(distribution_init_vz)
sigmavzi = np.sqrt(pvariance(distribution_init_vz))
distribution_init_vz.sort()
vzi = norm.pdf(distribution_init_vz, muvzi, sigmavzi)
ax1.plot(distribution_init_vz, vzi, 'r--', label = 'Normal('+str('%.2f' % muvzi)+','+str('%.4f' % sigmavzi)+')')
ax1.set_xlabel('vz initial')
#ax1.set_title('Initial velocity distribution along the z axis')
ax1.legend()

ax2 = plt.subplot(312)
nvyi, binsvyi, patchesvyi = ax2.hist(distribution_init_vy, density = True)
muvyi = mean(distribution_init_vy)
sigmavyi = np.sqrt(pvariance(distribution_init_vy))
distribution_init_vy.sort()
vyi = norm.pdf(distribution_init_vy, muvyi, sigmavyi)
ax2.plot(distribution_init_vy, vyi, 'r--',label = 'Normal('+str('%.4f' % muvyi)+','+str('%.3f' % sigmavyi)+')')
ax2.set_xlabel('vy initial')
#ax2.set_title('Initial velocity distribution along the y axis')
ax2.legend()

ax3 = plt.subplot(313)
nvxi, binsvxi, patchesvxi = ax3.hist(distribution_init_vx, density = True)
muvxi = mean(distribution_init_vx)
sigmavxi = np.sqrt(pvariance(distribution_init_vx))
distribution_init_vx.sort()
vxi = norm.pdf(distribution_init_vx, muvxi, sigmavxi)
ax3.plot(distribution_init_vx, vxi, 'r--', label = 'Normal('+str('%.4f' % muvxi)+','+str('%.3f' % sigmavxi)+')')
ax3.set_xlabel('vx initial')
#ax3.set_title('Initial velocity distribution along the x axis')
ax3.legend()

plt.subplots_adjust(hspace = 0.5)
plt.suptitle('Initial distribution of the speed of the cloud of '+str(nb_part)+' atoms')
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_init_v'+simulation_name+'.png')

#compute initial temperature
print("Initial temperature : - in all direction", mass/3/kb*(sigmavxi**2+sigmavyi**2+sigmavzi**2+muvzi**2+muvyi**2+muvxi**2))
print("- in the z direction : ", mass/kb*(sigmavzi**2+muvzi**2))
print("- in the y direction : ", mass/kb*(sigmavyi**2+muvyi**2))
print("- in the x direction :", mass/kb*(sigmavxi**2+muvxi**2))

#display the initial distribution in position
fig2 = plt.figure(2)

ax4 = plt.subplot(311)
ax4.hist(distribution_init_z, density = True)
muzi = mean(distribution_init_z)
sigmazi = np.sqrt(pvariance(distribution_init_z))
distribution_init_z.sort()
zi = norm.pdf(distribution_init_z, muzi, sigmazi)
ax4.plot(distribution_init_z, zi, 'r--', label = 'Normal('+str('%.7f' % muzi)+','+str('%.5f' % sigmazi)+')')
ax4.set_xlabel('z initial')
#ax4.set_title('Initial position distribution along the z axis')
ax4.legend()

ax5 = plt.subplot(312)
ax5.hist(distribution_init_y, density = True)
muyi = mean(distribution_init_y)
sigmayi = np.sqrt(pvariance(distribution_init_y))
distribution_init_y.sort()
yi = norm.pdf(distribution_init_y, muyi, sigmayi)
ax5.plot(distribution_init_y, yi, 'r--', label = 'Normal('+str('%.7f' % muyi)+','+str('%.5f' % sigmayi)+')')
ax5.set_xlabel('y initial')
#ax5.set_title('Initial position distribution along the y axis')
ax5.legend()

ax6 = plt.subplot(313)
ax6.hist(distribution_init_x, density = True)
muxi = mean(distribution_init_x)
sigmaxi = np.sqrt(pvariance(distribution_init_x))
distribution_init_x.sort()
xi = norm.pdf(distribution_init_x, muxi, sigmaxi)
ax6.plot(distribution_init_x, xi, 'r--', label = 'Normal('+str('%.7f' %muxi)+','+str('%.5f' %sigmaxi)+')')
ax6.set_xlabel('x initial')
#ax6.set_title('Initial position distribution along the x axis')
ax6.legend()

plt.subplots_adjust(hspace = 0.5)
plt.suptitle('Initial position distribution of the cloud of '+str(nb_part)+' atoms')
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_init_pos'+simulation_name+'.png')


#take the final distributions
distribution_final_z = [sol_arr[i].y[1,nb_iter-1]*L for i in range(nb_part)]
distribution_final_y = [sol_arr[i].y[3,nb_iter-1]*L for i in range(nb_part)]
distribution_final_x = [sol_arr[i].y[5,nb_iter-1]*L for i in range(nb_part)]
distribution_final_vz = [sol_arr[i].y[0,nb_iter-1]*vL for i in range(nb_part)]
distribution_final_vy = [sol_arr[i].y[2,nb_iter-1]*vL for i in range(nb_part)]
distribution_final_vx = [sol_arr[i].y[4,nb_iter-1]*vL for i in range(nb_part)]

#display the final distributions in speed
fig3 = plt.figure(3)

ax7 = plt.subplot(311)
nvzf, binsvzf, patchesvzf = ax7.hist(distribution_final_vz, density = True)
muvzf = mean(distribution_final_vz)
sigmavzf = np.sqrt(pvariance(distribution_final_vz))
distribution_final_vz.sort()
vzf = norm.pdf(distribution_final_vz, muvzf, sigmavzf)
ax7.plot(distribution_final_vz, vzf, 'r--',label = 'Normal('+str('%.2f' %muvzf)+','+str('%.4f' %sigmavzf)+')')
ax7.set_xlabel('vz final')
#ax7.set_title('Final speed distribution along the z axis')
ax7.legend()

ax8 = plt.subplot(312)
nvyf, binsvyf, patchesvyf = ax8.hist(distribution_final_vy, density = True)
muvyf = mean(distribution_final_vy)
sigmavyf = np.sqrt(pvariance(distribution_final_vy))
distribution_final_vy.sort()
vyf = norm.pdf(distribution_final_vy, muvyf, sigmavyf)
ax8.plot(distribution_final_vy, vyf, 'r--', label = 'Normal('+str('%.4f' %muvyf)+','+str('%.3f' %sigmavyf)+')')
ax8.set_xlabel('vy final')
#ax8.set_title('Final speed distribution along the y axis')
ax8.legend()

ax9 = plt.subplot(313)
nvxf, binsvxf, patchesvxf = ax9.hist(distribution_final_vx, density = True)
muvxf = mean(distribution_final_vx)
sigmavxf = np.sqrt(pvariance(distribution_final_vx))
distribution_final_vx.sort()
vxf = norm.pdf(distribution_final_vx, muvxf, sigmavxf)
ax9.plot(distribution_final_vx, vxf, 'r--', label = 'Normal('+str('%.4f' %muvxf)+','+str('%.3f' %sigmavxf)+')')
ax9.set_xlabel('vx final')
#ax9.set_title('Final speed distribution along the x axis')
ax9.legend()

plt.subplots_adjust(hspace = 0.5)
plt.suptitle('Final speed distribution of the cloud of ' + str(nb_part)+' atoms')
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_fin_v'+simulation_name+'.png')

#display the final distribution in space
fig4 = plt.figure(4)

ax10 = plt.subplot(311)
nzf, binszf, patcheszf = ax10.hist(distribution_final_z, density = True)
muzf = mean(distribution_final_z)
sigmazf = np.sqrt(pvariance(distribution_final_z))
distribution_final_z.sort()
zf = norm.pdf(distribution_final_z, muzf, sigmazf)
ax10.plot(distribution_final_z, zf, 'r--',label = 'Normal('+str('%.3f' %muzf)+','+str('%.4f' %sigmazf)+')')
ax10.set_xlabel('z final')
#ax10.set_title('Final position distribution along the z axis')
ax10.legend()


ax11 = plt.subplot(312)
num_bins = 100
muyf = mean(distribution_final_y)
sigmayf = np.sqrt(pvariance(distribution_final_y))
nyf, binsyf, patchesyf = ax11.hist(distribution_final_y, num_bins, label = 'sigma = '+ str('%.3f' %sigmayf))       
ax11.set_xlabel('y final')
#ax11.set_title('Final position distribution along the y axis')
ax11.legend()


ax12 = plt.subplot(313)
num_bins = 100
muxf = mean(distribution_final_x)
sigmaxf = np.sqrt(pvariance(distribution_final_x))
nxf, binsxf, patchesxf = ax12.hist(distribution_final_x, num_bins, label  = 'sigma ='+str('%.3f' % sigmaxf))
ax12.set_xlabel('x final')
#ax12.set_title('Final position distribution along the x axis')
ax12.legend()

plt.subplots_adjust(hspace = 0.5)
plt.suptitle('Final distribution of the position of the cloud of '+str(nb_part)+' atoms')
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_fin_pos'+simulation_name+'.png')

#filter the final set of particles according to their position
distribution_final_inside_vz = []
distribution_final_inside_vy = []
distribution_final_inside_vx = []
distribution_final_inside_z = []
distribution_final_inside_y = []
distribution_final_inside_x = []

for i in range(nb_part):
    if distribution_final_x[i]**2+distribution_final_y[i]**2<w0**2/2:
        distribution_final_inside_vz.append(distribution_final_vz[i])
        distribution_final_inside_vy.append(distribution_final_vy[i])
        distribution_final_inside_vx.append(distribution_final_vx[i])
        distribution_final_inside_z.append(distribution_final_z[i])
        distribution_final_inside_y.append(distribution_final_y[i])
        distribution_final_inside_x.append(distribution_final_x[i])

nb_part = len(distribution_final_inside_vz)
print("Nb of atoms inside the beam : ",nb_part)

distribution_final_filtered_vz = []
distribution_final_filtered_vy = []
distribution_final_filtered_vx = []
distribution_final_filtered_z = []
distribution_final_filtered_y = []
distribution_final_filtered_x = []

for i in range(nb_part):
    if 1/2*mass*(distribution_final_inside_vy[i]**2+distribution_final_inside_vx[i]**2) < Umax:
        distribution_final_filtered_vz.append(distribution_final_inside_vz[i])
        distribution_final_filtered_vy.append(distribution_final_inside_vy[i])
        distribution_final_filtered_vx.append(distribution_final_inside_vx[i])
        distribution_final_filtered_z.append(distribution_final_inside_z[i])
        distribution_final_filtered_y.append(distribution_final_inside_y[i])
        distribution_final_filtered_x.append(distribution_final_inside_x[i])

nb_part = len(distribution_final_filtered_vz)
print("Nb of atoms inside the beam : ",nb_part)

if nb_part>0:
    #display the final speed distribution of the filtered set 
    fig5 = plt.figure(5)
    
    ax13 = plt.subplot(311)
    nvzf, binsvzf, patchesvzf = ax13.hist(distribution_final_filtered_vz, density = True)
    muvzf = mean(distribution_final_filtered_vz)
    sigmavzf = np.sqrt(pvariance(distribution_final_filtered_vz))
    distribution_final_vz.sort()
    vzf = norm.pdf(distribution_final_filtered_vz, muvzf, sigmavzf)
    ax13.plot(distribution_final_filtered_vz, vzf, 'r--',label = 'Normal('+str('%.2f' %muvzf)+','+str('%.4f' %sigmavzf)+')')
    ax13.set_xlabel('vz final')
    #ax13.set_title('Final speed distribution along the z axis')
    ax13.legend()
    
    
    ax14 = plt.subplot(312)
    nvyf, binsvyf, patchesvyf = ax14.hist(distribution_final_filtered_vy, density = True)
    muvyf = mean(distribution_final_filtered_vy)
    sigmavyf = np.sqrt(pvariance(distribution_final_filtered_vy))
    distribution_final_filtered_vy.sort()
    vyf = norm.pdf(distribution_final_filtered_vy, muvyf, sigmavyf)
    ax14.plot(distribution_final_filtered_vy, vyf, 'r--', label = 'Normal('+str('%.4f' %muvyf)+','+str('%.3f' %sigmavyf)+')')
    ax14.set_xlabel('vy final')
    #ax14.set_title('Final speed distribution along the y axis')
    ax14.legend()
    
    ax15 = plt.subplot(313)
    nvxf, binsvxf, patchesvxf = ax15.hist(distribution_final_filtered_vx, density = True)
    muvxf = mean(distribution_final_filtered_vx)
    sigmavxf = np.sqrt(pvariance(distribution_final_filtered_vx))
    distribution_final_filtered_vx.sort()
    vxf = norm.pdf(distribution_final_filtered_vx, muvxf, sigmavxf)
    ax15.plot(distribution_final_filtered_vx, vxf, 'r--', label = 'Normal('+str('%.4f' %muvxf)+','+str('%.3f' %sigmavxf)+')')
    ax15.set_xlabel('vx final')
    #ax15.set_title('Final speed distribution along the x axis')
    ax15.legend()
    
    plt.subplots_adjust(hspace = 0.5)
    plt.suptitle('Final speed distribution of the cloud of '+str(nb_part)+' atoms')
    plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_fin_v_filtered'+simulation_name+'.png')
    
    #compute final temperature
    print("Final temperature : - in 3 direction :", mass/6/kb*(sigmavxf**2+sigmavyf**2+sigmavzf**2+muvzf**2+muvyf**2+muvxf**2))
    print("- along the z axis : ",mass/kb*(sigmavzf**2+muvzf**2))
    print("- along the y axis : ", mass/kb*(sigmavyf**2+muvyf**2))
    print("- along the x axis : ", mass/kb*(sigmavxf**2+muvxf**2))
    
    # Display the final space distribution of the filtered set
    fig6 = plt.figure(6)
    
    ax16 = plt.subplot(311)
    num_bins = 25
    nzf, binszf, patcheszf = ax16.hist(distribution_final_filtered_z, num_bins, density = True)
    muzf = mean(distribution_final_filtered_z)
    sigmazf = np.sqrt(pvariance(distribution_final_filtered_z))
    distribution_final_filtered_z.sort()
    zf = norm.pdf(distribution_final_filtered_z, muzf, sigmazf)
    ax16.plot(distribution_final_filtered_z, zf, 'r--', label = 'Normal('+str('%.3f' %muzf)+','+str('%.4f' %sigmazf)+')')
    ax16.set_xlabel('z final')
    #ax16.set_title('Final position distribution along the z axis')
    ax16.legend()

    
    ax17 = plt.subplot(312)
    num_bins = 30
    nyf, binsyf, patchesyf = ax17.hist(distribution_final_filtered_y, num_bins, density = True)
    muyf = mean(distribution_final_filtered_y)
    sigmayf = np.sqrt(pvariance(distribution_final_filtered_y))
    distribution_final_filtered_y.sort()
    yf = norm.pdf(distribution_final_filtered_y, muyf, sigmayf)
    ax17.plot(distribution_final_filtered_y, yf, 'r--', label = 'Normal('+str('%.7f' %muyf)+','+str('%.5f' %sigmayf)+')')
    ax17.set_xlabel('y final')
    #ax17.set_title('Final position distribution along the y axis')
    ax17.legend()
    
    ax18 = plt.subplot(313)
    num_bins = 30
    nxf, binsxf, patchesxf = ax18.hist(distribution_final_filtered_x, num_bins, density = True)
    muxf = mean(distribution_final_filtered_x)
    sigmaxf = np.sqrt(pvariance(distribution_final_filtered_x))
    distribution_final_filtered_x.sort()
    xf = norm.pdf(distribution_final_filtered_x, muxf, sigmaxf)
    ax18.plot(distribution_final_filtered_x, xf, 'r--',label = 'Normal('+str('%.7f' %muxf)+','+str('%.5f' %sigmaxf)+')')
    ax18.set_xlabel('x final')
    #ax18.set_title('Final position distribution along the x axis')
    ax18.legend()
    
    plt.subplots_adjust(hspace = 0.5)
    plt.suptitle('Final position distribution of the cloud of '+str(nb_part)+' atoms')
    plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/distr_fin_pos_filtered'+simulation_name+'.png')

plt.show()