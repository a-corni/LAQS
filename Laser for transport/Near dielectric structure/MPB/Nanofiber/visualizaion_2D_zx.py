import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors
plt.close('all')
from scipy import constants as cst
import cesium_data as cs 
#import cesium_data as cs

SX = 8.0
SY = SX
SZ = SX
kz = 1.013197483552876*2.0*np.pi/1.0e-6 #in rad.m-1

# set the resolution higher to 512 to check that beta is closer to analytic solution 

#print('beta:', 1.342308553122232*2.0*np.pi/1.0e-6) 
#1.342308553122232 

#-------------------------------------
# #load relative electric permittivity  
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-epsilon.780.h5'  
data = h5py.File(filename,'r') 

eps = data['data'][:,:] 
rindex = np.sqrt(eps) # get the refractive index from relative permittivity epsilon_r = n^2 
print(eps.shape)

grid_spacing = SX/eps.shape[0] #um

print('grid spacing (um)', grid_spacing)
resolution = eps.shape[0]/SX  #nbpixels/um
print('resolution', resolution)

#-------------------------------------
#plt.figure(1)
#load E-field data for 780 nm light
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-e.k01.b01.yodd.780.h5'  
data_780 = h5py.File(filename,'r') 

#keys = list(data.keys()) 
#print(keys) 

Ex_780 = data_780['x.r'][:,:] + 1j*data_780['x.i'][:,:] #um.kg.s-3.A-1 
Ey_780 = data_780['y.r'][:,:] + 1j*data_780['y.i'][:,:] 
Ez_780 = data_780['z.r'][:,:] + 1j*data_780['z.i'][:,:] 

Ix_780 = np.array([np.real(E*np.conj(E)) for E in Ex_780]) #um².kg².s-6.A-2 
Iy_780 = np.array([np.real(E*np.conj(E)) for E in Ey_780]) 
Iz_780 = np.array([np.real(E*np.conj(E)) for E in Ez_780]) 

#total electric field intensity (NOT light intensity)
It_780 = Ix_780+Iy_780+Iz_780 

#load Poynting data and energy density to compute group velocity
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-flux.v.k01.b01.z.yodd.780.h5'  
data_780_p = h5py.File(filename,'r') 
Sz_780 = 0.5*data_780_p['z.r'][:,:] # time-average flux density (1/2 of the real part of complex poynting vector returned by MPB 
#careful I got it wrong in terms of computing the surface element value.. surface/(nbgridpoint1x*nbgridpoint1y) which is not 128 but 256.. 128 is the resolution 
#power P => integrate over entire transverse plane 
surface_element = (grid_spacing)**2#grid_spacing**2 #um**2

P_780 = np.sum(Sz_780)*surface_element #kg.um².s-3 
print('Power in kg.(um)².s-3', P_780) 

#load total energy density 
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-tot.rpwr.k01.b01.yodd.780.h5'  
data_780_e = h5py.File(filename,'r') 

utot_780 = 2.0/SX*data_780_e['data'][:,:] #kg.um².s-3/um²# cycle-average total EM energy density  

#integrate over cell (total energy) 
Utot_780 = np.sum(utot_780)*surface_element #kg.um².s-3
print("Total energy density in kg.(um)².s-2", Utot_780) 

vg_780 = P_780/Utot_780 #unitary 

print('vg',vg_780) 

#in SI units now 
vg_780 = vg_780*cst.c #m.s-1

#-------------------------------------
# normalize fields

P_780 = vg_780*cst.epsilon_0*0.5*np.sum(np.multiply(It_780,eps))*(SX*1.0e-6*SY*1.0e-6)/(It_780.shape[0]*It_780.shape[1]) #kg.m².s-3

Power_Wanted_780 = 12e-3 #Watt # nanofiber exp = 8 mW
Cnorm_780 = Power_Wanted_780/P_780 

I_780 = It_780*Cnorm_780 #W.m-2
m = I_780.shape[1]
xp = np.linspace(-SX/2, SX/2, I_780.shape[0], endpoint=False) 
zp = np.linspace(-SZ/2, SZ/2, m, endpoint=False)
Ex_zx_780 = np.array([[Ex_780[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)])#um.kg.s-3.A-1 
Ey_zx_780 = np.array([[Ey_780[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)])
Ez_zx_780 = np.array([[Ez_780[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)])#pi shift
Ix_zx_780 = np.array([np.real(E*np.conj(E)) for E in Ex_zx_780]) #um².kg².s-6.A-2 
Iy_zx_780 = np.array([np.real(E*np.conj(E)) for E in Ey_zx_780]) 
Iz_zx_780 = np.array([np.real(E*np.conj(E)) for E in Ez_zx_780]) 

#total electric field intensity (NOT light intensity) 
I_zx_780 = Cnorm_780*(Ix_zx_780+Iy_zx_780+Iz_zx_780)
impedance = np.sqrt(cst.mu_0/cst.epsilon_0)
#Ilight_780 = I_780/(2.0*impedance)
#Itmax_780 = np.max(It_780) 



# plt.xlabel('y (um)') 
# plt.ylabel('x (um)') 
# plt.xlim([-SX/2,SX/2]) 
# plt.ylim([-SY/2,SY/2]) 
# 
# plt.title('780nm light-field intensity transverse profile') 
# plt.imshow(Ilight_780,cmap=plt.cm.bone,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
# cbar = plt.colorbar() 
# cbar.set_label('Light intensity [in W.m-2]',rotation=270,labelpad=20)
# plt.tight_layout()
# plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/780_light_intensity.png')

#-------------------------------------
#load E-field data for 1057 nm light
#-------------------------------------
# plt.figure(2)

filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-e.k01.b01.yeven.1057.h5'  
data_1057 = h5py.File(filename,'r') 

#keys = list(data.keys()) 
#print(keys) 

Ex_1057 = data_1057['x.r'][:,:] + 1j*data_1057['x.i'][:,:] #um.kg.s-3.A-1 
Ey_1057 = data_1057['y.r'][:,:] + 1j*data_1057['y.i'][:,:] 
Ez_1057 = data_1057['z.r'][:,:] + 1j*data_1057['z.i'][:,:] 

Ix_1057 = np.array([np.real(E*np.conj(E)) for E in Ex_1057])#um².kg².s-6.A-2  
Iy_1057 = np.array([np.real(E*np.conj(E)) for E in Ey_1057]) 
Iz_1057 = np.array([np.real(E*np.conj(E)) for E in Ez_1057]) 

#total electric field intensity (NOT light intensity)
It_1057 = Ix_1057+Iy_1057+Iz_1057 

#load Poynting data and energy density to compute group velocity
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-flux.v.k01.b01.z.yeven.1057.h5'  
data_1057_p = h5py.File(filename,'r') 
Sz_1057 = 0.5*data_1057_p['z.r'][:,:] # time-average flux density (1/2 of the real part of complex poynting vector returned by MPB 

#careful I got it wrong in terms of computing the surface element value.. surface/(nbgridpoint1x*nbgridpoint1y) which is not 128 but 256.. 128 is the resolution 
#power P => integrate over entire transverse plane 

P_1057 = np.sum(Sz_1057)*surface_element 
print('MPB power', P_1057) 

#load total energy density 
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-tot.rpwr.k01.b01.yeven.1057.h5'  
data_1057_e = h5py.File(filename,'r') 
utot_1057 = 2.0/SX*data_1057_e['data'][:,:] # cycle-average total EM energy density 

#integrate over cell (total energy) 
Utot_1057 = np.sum(utot_1057)*surface_element 
print(Utot_1057) 

vg_1057 = P_1057/Utot_1057 

print('1057nm : vg',vg_1057) 

#in SI units now 
vg_1057 = vg_1057*cst.c 

#-------------------------------------
# normalize field

P_1057 = vg_1057*cst.epsilon_0*0.5*np.sum(np.multiply(It_1057,eps))*(SX*1.0e-6*SY*1.0e-6)/(It_1057.shape[0]*It_1057.shape[1]) 

Power_Wanted_1057 = 1.1e-3 #Watt  nanofiber experiment = 1.5 mW
Cnorm_1057 = Power_Wanted_1057/P_1057 

I_1057 = It_1057*Cnorm_1057
xp = np.linspace(-SX/2,SX/2,I_1057.shape[0],endpoint=False) 
zp = np.linspace(-SY/2,SY/2,I_1057.shape[1],endpoint=False) 

# plt.xlabel('y (um)') 
# plt.ylabel('x (um)') 
# plt.xlim([-SX/2,SX/2]) 
# plt.ylim([-SY/2,SY/2]) 
# 
# plt.title('1057nm e-field intensity transverse profile') 
# plt.imshow(I_1057,cmap=plt.cm.hot,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
# plt.savefig('1beam_1057_Electric_intensity.png')

#-------------------------------------
#plt.figure(3)
# now we compute a standing-wave red fields (imagine retro-reflection on a mirror)
# at a given z (we skip the time phase omega*t)
# z  = 0.0 #in m
kz = 1.013197483552876*2.0*np.pi/1.0e-6 #in rad.m-1
# 
# Exs = Ex_1057*np.exp(-1j*kz*z)+Ex_1057*np.exp(1j*kz*z) #um.kg.s-3.A-1 
# Eys = Ey_1057*np.exp(-1j*kz*z)+Ey_1057*np.exp(1j*kz*z)
# Ezs = Ez_1057*np.exp(-1j*kz*z)-Ez_1057*np.exp(1j*kz*z)  #pi shift
# 
# Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
# Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
# Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 
# 
# Its = Ixs+Iys+Izs
# Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 

#-----------
# Isat units
#Isat = 2.71
#impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
#print('Imax/Isat', 0.1*Itmax/(2.0*impedance)/Isat) # in mW/cm2 
#It = 0.1*It/(2.0*impedance)/Isat # in mW/cm2 
 
m = I_1057.shape[1]
xp = np.linspace(-SX/2, SX/2, I_1057.shape[0], endpoint=False) 
zp = np.linspace(-SZ/2, SZ/2, m, endpoint=False)
Ex_zx_1057 = np.array([[Ex_1057[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6)+Ex_1057[i,m//2]*np.exp(1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)])#um.kg.s-3.A-1 
Ey_zx_1057 = np.array([[Ey_1057[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6)+Ey_1057[i,m//2]*np.exp(1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)])
Ez_zx_1057 = np.array([[Ez_1057[i,m//2]*np.exp(-1j*kz*zp[j]*1e-6)-Ez_1057[i,m//2]*np.exp(1j*kz*zp[j]*1e-6) for j in range(m)] for i in range(m)]) #pi shift

print(Ey_zx_1057[m//2,m//2])
print(Ex_zx_1057[m//2,m//2])
Ix_zx_1057 = np.array([np.real(E*np.conj(E)) for E in Ex_zx_1057]) #um².kg².s-6.A-2 
Iy_zx_1057 = np.array([np.real(E*np.conj(E)) for E in Ey_zx_1057]) 
Iz_zx_1057 = np.array([np.real(E*np.conj(E)) for E in Ez_zx_1057]) 

#total electric field intensity (NOT light intensity) 
I_zx_1057 = Cnorm_1057*(Ix_zx_1057+Iy_zx_1057+Iz_zx_1057)
impedance = np.sqrt(cst.mu_0/cst.epsilon_0)# plt.xlabel('y (um)') 
# plt.ylabel('x (um)') 
# plt.xlim([-SX/2,SX/2]) 
# plt.ylim([-SY/2,SY/2]) 
# 
# plt.title('Counter-propagating 1057nm light-field intensity transverse profile') 

# impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
# # Isat = 2.71e-3/(1e-2)**2 # in W/m^2 #2.71 # mW/cm2
# Islight_1057 = Is_1057/(2.0*impedance) #W.m-2
# # Ib_sat = 0.1*Ib/Isat # I_blue in units of Isat (convert first in mW/cm2) and then light intensi vac imp
# 
# plt.imshow(Islight_1057,cmap=plt.cm.hot,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
# cbar = plt.colorbar() 
# cbar.set_label('Light Intensity [in W.m-2]',rotation=270,labelpad=20)
# plt.tight_layout()
# plt.savefig('1057_light_intensity.png')
# cmap = plt.cm.hot
# norm = plt.Normalize()
# image = cmap(norm(Ib))
#plt.imsave('1057_light_intensity.png',image)  
#   
# cbar = plt.colorbar() 
# cbar.set_label('I in (Isat)',rotation=270,labelpad=20)
# plt.tight_layout()


# single atom resonant optical depth as a function of intensity/position (take into account saturation)
# sigma_0 = 0.14319e-12 # m^2 # must be consistent with Isat (i.e) choice of polarization and transition
# alpha0 = sigma_0/Aeff = sigma0*I(r)/Ptot
# alpha = alpha0/(1+I/Isat)
# alpha = sigma0*I/Ptot/(1.0+I/Isat)
# alpha = (sigma_0/Power_Wanted_blue)*np.divide(Ib,(1.0+Ib/Isat))

# plt.figure(2) 
# 
# xp = np.linspace(-SX/2.0,SX/2.0,I_blue.shape[0],endpoint=False) 
# yp = np.linspace(-SY/2.0,SY/2.0,I_blue.shape[1],endpoint=False) 
# 
# plt.xlabel('y (um)') 
# plt.ylabel('x (um)')
# plt.xlim([-0.5,0.5]) 
# plt.ylim([-0.5,0.5]) 
# 
# plt.title('single-atom resonant optical depth profile (P= %2.2g pW)' % (Power_Wanted_blue*1.0e12)) 
# 
# 
# plt.imshow(alpha,cmap=plt.cm.hot,vmin=0.0,vmax=np.max(alpha),extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
#   
# cbar = plt.colorbar() 
# cbar.set_label('OD',rotation=270,labelpad=20)
# plt.tight_layout()
# plt.savefig('optical_depth.png')  

#-------------------------------
plt.figure(4)
#compute the potential of the dipole trap
print("compute potential ...")
#-------------------------------

kb = 1.3806504e-23 # J/K # Boltzmann constant
wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
gamma = 2*np.pi*5.234e6 #Hz #decay rate
delta_1057 = 2*np.pi/1057*1e9*cst.c-wa
delta_780 = 2*np.pi/780*1e9*cst.c-wa

print("detuning 1057", delta_1057,"detuning 780", delta_780)

def potential(fieldint,wavelength):
    #fieldint is the field intensity |E|^2.. no impedance involved etc..
    return -1.0/4.0*cs.alphaComplet(wavelength)*fieldint

#Nanofiber_potential = 3*np.pi*cst.c**2/2/wa**3*gamma*(1/delta_1057*Islight_1057+1/delta_780*Ilight_780)
Nanofiber_potential = potential(I_zx_1057, 1057.0e-9) +potential(I_zx_780,780.0e-9)
xp = np.linspace(-SX/2,SX/2,I_zx_1057.shape[0],endpoint=False) 
zp = np.linspace(-SY/2,SY/2,I_zx_1057.shape[1],endpoint=False) 
r0 = 0.235

for i,x in enumerate(xp):
    for j,y in enumerate(zp):
                
        if np.abs(xp[i]) < r0:
            Nanofiber_potential[i,j] = 0
        else:
            Nanofiber_potential[i,j] -=  (5.6E-49)/(((np.abs(xp[i])-r0)*1e-6)**3) #add vanderwals potential

#plot 2D image of the potential
plt.figure(1)

plt.xlabel('z (um)') 
plt.ylabel('x (um)')
plt.xlim([-1,1]) 
plt.ylim([-1.5,1.5]) 

levels = [-2.5,-2,-1.5,-1,-0.5,0,1,2,3,4,5]
plt.title('Trap potential nanofiber experiment') 
plt.imshow(Nanofiber_potential/kb*1e3, norm = colors.DivergingNorm(vmin=-0.2, vcenter=0, vmax=1.0), cmap=plt.cm.RdYlBu,extent=[xp[0],xp[-1],zp[0],zp[-1]])#,interpolation='none') 
cbar = plt.colorbar() 
cbar.set_label('Trap Potential [in mK]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_experiment/Nanofiber_Trap_potential.png')

# plot 2D projection of the atoms
# back of all images

#3D trajectories
simulation_name = "Nanofiber_experiment"
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_experiment/solverp0_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']

nb_part = sol_arr.shape[0]
nb_iter = len(sol_arr[0].t)
tend = 10e-5
dt = 0.0025*tend # 0.25% of simulation time

for i in range(0, nb_iter, 10):
    
    plt.figure(i+3)
    
    px = [sol_arr[j].y[5,i]*1e6 for j in range(nb_part)]
    pz = [sol_arr[j].y[1,i]*1e6 for j in range(nb_part)]
    
    plt.xlabel('z (um)') 
    plt.ylabel('x (um)') 
    plt.xlim([-1,1])
    plt.ylim([-1.5,1.5]) 
    
    plt.title('Simulation t = '+ '%1.f'%(i*dt*1e6)+ ' µs') 
    plt.imshow(Nanofiber_potential/kb*1e3, norm = colors.DivergingNorm(vmin=-0.2, vcenter=0, vmax=1.0), cmap=plt.cm.RdYlBu,extent=[zp[0],zp[-1],xp[0],xp[-1]])#,interpolation='none')
    cbar = plt.colorbar() 
    plt.scatter(pz, px, s = 0.1)
    cbar.set_label('Trap Potential [in mK]',rotation=270,labelpad=20)
    plt.tight_layout()
    plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_experiment/Nanofiber_Trap_potential_zx'+str(i)+'.png')
    