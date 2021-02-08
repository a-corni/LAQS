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
# set the resolution higher to 512 to check that beta is closer to analytic solution 

#print('beta:', 1.342308553122232*2.0*np.pi/1.0e-6) 
#1.342308553122232 

#-------------------------------------
# #load relative electric permittivity  
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-epsilon.780.h5'  
data = h5py.File(filename,'r') 

eps = data['data'][:,:] 
rindex = np.sqrt(eps) # get the refractive index from relative permittivity epsilon_r = n^2 
print(eps.shape)

grid_spacing = SX/eps.shape[0] #um

print('grid spacing (um)', grid_spacing)
resolution = eps.shape[0]/SX  #nbpixels/um
print('resolution', resolution)

#-------------------------------------
plt.figure(1)
#load E-field data for 780 nm light
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-e.k01.b01.yodd.780.h5'  
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
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-flux.v.k01.b01.z.yodd.780.h5'  
data_780_p = h5py.File(filename,'r') 
Sz_780 = 0.5*data_780_p['z.r'][:,:] # time-average flux density (1/2 of the real part of complex poynting vector returned by MPB 
#careful I got it wrong in terms of computing the surface element value.. surface/(nbgridpoint1x*nbgridpoint1y) which is not 128 but 256.. 128 is the resolution 
#power P => integrate over entire transverse plane 
surface_element = (grid_spacing)**2#grid_spacing**2 #um**2

P_780 = np.sum(Sz_780)*surface_element #kg.um².s-3 
print('Power in kg.(um)².s-3', P_780) 

#load total energy density 
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-tot.rpwr.k01.b01.yodd.780.h5'  
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

Power_Wanted_780 = 8e-3 #Watt # nanofiber exp = 8 mW
Cnorm_780 = Power_Wanted_780/P_780 

I_780 = It_780*Cnorm_780 #W.m-2

impedance = np.sqrt(cst.mu_0/cst.epsilon_0)
Ilight_780 = I_780/(2.0*impedance)
#Itmax_780 = np.max(It_780) 


xp = np.linspace(-SX/2,SX/2,I_780.shape[0],endpoint=False) 
yp = np.linspace(-SY/2,SY/2,I_780.shape[1],endpoint=False) 

plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-SX/2,SX/2]) 
plt.ylim([-SY/2,SY/2]) 

plt.title('780nm light-field intensity transverse profile') 
plt.imshow(Ilight_780,cmap=plt.cm.bone,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
cbar = plt.colorbar() 
cbar.set_label('Light intensity [in W.m-2]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_780_light_intensity.png')

#-------------------------------------
#load E-field data for 1057 nm light
#-------------------------------------
plt.figure(2)

filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-e.k01.b01.yeven.1057.h5'  
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
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-flux.v.k01.b01.z.yeven.1057.h5'  
data_1057_p = h5py.File(filename,'r') 
Sz_1057 = 0.5*data_1057_p['z.r'][:,:] # time-average flux density (1/2 of the real part of complex poynting vector returned by MPB 

#careful I got it wrong in terms of computing the surface element value.. surface/(nbgridpoint1x*nbgridpoint1y) which is not 128 but 256.. 128 is the resolution 
#power P => integrate over entire transverse plane 

P_1057 = np.sum(Sz_1057)*surface_element 
print('MPB power', P_1057) 

#load total energy density 
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam-tot.rpwr.k01.b01.yeven.1057.h5'  
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

Power_Wanted_1057 = 3e-3 #Watt  nanofiber experiment = 1.5 mW
Cnorm_1057 = Power_Wanted_1057/P_1057 

I_1057 = It_1057*Cnorm_1057
xp = np.linspace(-SX/2,SX/2,I_1057.shape[0],endpoint=False) 
yp = np.linspace(-SY/2,SY/2,I_1057.shape[1],endpoint=False) 

plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-SX/2,SX/2]) 
plt.ylim([-SY/2,SY/2]) 

plt.title('1057nm e-field intensity transverse profile') 
plt.imshow(I_1057,cmap=plt.cm.hot,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_1beam_1057_Electric_intensity.png')

#-------------------------------------
plt.figure(3)
# now we compute a standing-wave red fields (imagine retro-reflection on a mirror)
# at a given z (we skip the time phase omega*t)
z  = 0.0 #in m
kz = 1.013197483552876*2.0*np.pi/1.0e-6 #in rad.m-1

Exs = Ex_1057*np.exp(-1j*kz*z)+Ex_1057*np.exp(1j*kz*z) #um.kg.s-3.A-1 
Eys = Ey_1057*np.exp(-1j*kz*z)+Ey_1057*np.exp(1j*kz*z)
Ezs = Ez_1057*np.exp(-1j*kz*z)-Ez_1057*np.exp(1j*kz*z)  #pi shift

Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 

Its = Ixs+Iys+Izs
Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 

#-----------
# Isat units
#Isat = 2.71
#impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
#print('Imax/Isat', 0.1*Itmax/(2.0*impedance)/Isat) # in mW/cm2 
#It = 0.1*It/(2.0*impedance)/Isat # in mW/cm2 
 
xp = np.linspace(-SX/2,SX/2,Is_1057.shape[0],endpoint=False) 
yp = np.linspace(-SY/2,SY/2,Is_1057.shape[1],endpoint=False) 

plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-SX/2,SX/2]) 
plt.ylim([-SY/2,SY/2]) 

plt.title('Counter-propagating 1057nm light-field intensity transverse profile') 

impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
# Isat = 2.71e-3/(1e-2)**2 # in W/m^2 #2.71 # mW/cm2
Islight_1057 = Is_1057/(2.0*impedance) #W.m-2
# Ib_sat = 0.1*Ib/Isat # I_blue in units of Isat (convert first in mW/cm2) and then light intensi vac imp

plt.imshow(Islight_1057,cmap=plt.cm.hot,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
cbar = plt.colorbar() 
cbar.set_label('Light Intensity [in W.m-2]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_1057_light_intensity.png')
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
Nanofiber_potential = potential(Is_1057, 1057.0e-9) +potential(I_780,780.0e-9)
xp = np.linspace(-SX/2,SX/2,Is_1057.shape[0],endpoint=False) 
yp = np.linspace(-SY/2,SY/2,Is_1057.shape[1],endpoint=False) 
r0 = 0.235

for i,x in enumerate(xp):
    for j,y in enumerate(yp):
        testx = (abs(xp[i])<r0)
        testy = (abs(yp[j])<r0)
        if testx and testy:
            Nanofiber_potential[i,j] = 0
        else:
            if testx:
                Nanofiber_potential[i,j] -=  (5.6E-49)/(((abs(yp[j])-r0)*1e-6)**3) #add vanderwals
            elif testy:
                Nanofiber_potential[i,j] -=  (5.6E-49)/(((abs(xp[i])-r0)*1e-6)**3) #add vanderwals potential
            else :
                Nanofiber_potential[i,j] -=  (5.6E-49)/(np.sqrt(((abs(xp[i])-r0)*1e-6)**2 +((abs(yp[j])-r0)*1e-6)**2)**3)

#plot 2D image of the potential
plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-1.0,1.0]) 
plt.ylim([-1.0,1.0]) 

levels = [-2.5,-2,-1.5,-1,-0.5,0,1,2,3,4,5]
plt.title('Trap potential nanofiber experiment') 
plt.imshow(Nanofiber_potential/kb*1e3, norm = colors.DivergingNorm(vmin=-1.0, vcenter=0, vmax=1.0), cmap=plt.cm.RdYlBu,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
cbar = plt.colorbar() 
cbar.set_label('Trap Potential [in mK]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_Trap_potential.png')

#-----------------------------------
#compute the gradient of the potential
#get the dipole force
print("compute gradient...")
#-----------------------------------
def grad_field(I):
    
    size = np.shape(I)
    
    grad_I_x = np.zeros(size)
    grad_I_x[:,0] = (I[:,1]-I[:,-1])/(2*grid_spacing)
    grad_I_x[:,size[1]-1] = (I[:,0]-I[:,-2])/(2*grid_spacing)
    for i in range(1,size[1]-1):
        grad_I_x[:,i] = (I[:,i+1]-I[:,i-1])/(2*grid_spacing) 
    
    grad_I_y = np.zeros(size)
    grad_I_y[0,:] = (I[1,:]-I[-1,:])/(2*grid_spacing)
    grad_I_y[size[0]-1,:] = (I[0,:]-I[-2,:])/(2*grid_spacing)
    for i in range(1,size[0]-1):
        grad_I_y[i,:] = (I[i+1,:]-I[i-1,:])/(2*grid_spacing)
    
    return np.array([grad_I_x*1e6,grad_I_y*1e6])

grad_It_780 = grad_field(Ilight_780)
grad_It_1057 = grad_field(Islight_1057)
grad_U0 = grad_field(Nanofiber_potential)

plt.figure(5)

xp = np.linspace(-SX/2, SX/2, Nanofiber_potential.shape[0], endpoint=False) 
yp = np.linspace(-SY/2, SY/2, Nanofiber_potential.shape[1],endpoint=False) 
plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-SX/2,SX/2]) 
plt.ylim([-SY/2,SY/2]) 

grad_U0_x_display = grad_U0[0]
grad_U0_y_display = grad_U0[1]

for i in range(Nanofiber_potential.shape[0]):
    for j in range(Nanofiber_potential.shape[1]):
        if (abs(xp[i])<r0) and (abs(yp[j])<r0):
            grad_U0_x_display[i,j] = 0
            grad_U0_y_display[i,j] = 0


levels = [-2.5,-2,-1.5,-1,-0.5,0,1,2,3,4,5]
plt.title('x composant gradient of potential for nanofiber experiment') 
plt.imshow(grad_U0_x_display, cmap=plt.cm.RdYlBu,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
cbar = plt.colorbar(spacing="proportional") 
cbar.set_label('grad potential along x[in N]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_Trap_force_x.png')

plt.figure(6)

xp = np.linspace(-SX/2, SX/2, Nanofiber_potential.shape[0], endpoint=False) 
yp = np.linspace(-SY/2, SY/2, Nanofiber_potential.shape[1], endpoint=False) 
plt.xlabel('y (um)') 
plt.ylabel('x (um)') 
plt.xlim([-SX/2,SX/2]) 
plt.ylim([-SY/2,SY/2]) 

levels = [-2.5,-2,-1.5,-1,-0.5,0,1,2,3,4,5]
plt.title('y composant gradient of potential for nanofiber experiment') 
plt.imshow(grad_U0_y_display, cmap=plt.cm.RdYlBu,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
cbar = plt.colorbar(spacing="proportional") 
cbar.set_label('grad potential along y [in N]',rotation=270,labelpad=20)
plt.tight_layout()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_Trap_force_y.png')

#--------------------------------------------
#Computation of the interpolation function
print("compute cuts...")
#--------------------------------------------


def interpolate(field, x, y):
    
    #provide y, x in m
    #bilinear interpolation
    
    ystar = y+4*1e-6 #m
    xstar = x+4*1e-6 #m
    ix = xstar/grid_spacing/1e-6 #index
    iy = ystar/grid_spacing/1e-6 #index
    x1 = int(ix)
    y1 = int(iy)
    dxstar = ix-x1
    dystar = iy-y1
    
    (field1, field2, field3, field4) = (field[x1%field.shape[0],y1%field.shape[1]], field[(x1+1)%field.shape[0],y1%field.shape[1]], field[x1%field.shape[0],(y1+1)%field.shape[1]],  field[(x1+1)%field.shape[0], (y1+1)%field.shape[1]])
        
    return (field2 - field1)*dxstar + (field3-field1)*dystar+(field1+field4-field2-field3)*dxstar*dystar + field1

xp = np.linspace(-7*SX/8, 7*SX/8, 10*Nanofiber_potential.shape[1], endpoint=False) 
yp = np.linspace(-7*SY/8, 7*SY/8, Nanofiber_potential.shape[0], endpoint=False) 
#780nm*
Pot_780 = potential(I_780, 780e-9)
Pot_x_780 = np.array([interpolate(Pot_780, x*1e-6, 0) for x in xp])
Pot_y_780 = np.array([interpolate(Pot_780, 0, y*1e-6) for y in yp])
#1057nm
Pot_1057 = potential(I_1057, 1057e-9)
Pot_x_1057 = np.array([interpolate(Pot_1057, x*1e-6, 0) for x in xp])
Pot_y_1057 = np.array([interpolate(Pot_1057, 0, y*1e-6) for y in yp])
#Van der Waals
#Total
Pot_x = np.array([interpolate(Nanofiber_potential, x*1e-6, 0) for x in xp])
Pot_y = np.array([interpolate(Nanofiber_potential, 0, y*1e-6) for y in yp])
#yp = np.linspace(-SY/8*1e-6, SY/8*1e-6, Nanofiber_potential.shape[1], endpoint=False) 

print("cuts along x...")
plt.figure(7)

plt.xlabel('x (um)') 
plt.ylabel('Pot (in mK)') 
plt.xlim([-SX/2,SX/2])
plt.ylim([-10,30]) 

plt.title('Potential along x') 
#Nanofiber_potential_x = np.array([Nanofiber_potential, 0, x] for x in xp])
print("plot1")
plt.plot(xp, Pot_x_780/kb*1e3, label = "780nm")
print("plot2")
plt.plot(xp, Pot_x_1057/kb*1e3, label = "1057nm")
print("plot3")
plt.plot(xp, Pot_x/kb*1e3, label = "total")#,interpolation='none') 
print("plot4")
plt.legend()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_potential_x.png')

plt.figure(8)

print("cuts along y...")
plt.xlabel('y (um)') 
plt.ylabel('Pot (in mK)') 
plt.xlim([-SX/2,SX/2])
plt.ylim([-10,30]) 

plt.title('Potential along y') 
#Nanofiber_potential_x = np.array([Nanofiber_potential, 0, x] for x in xp])
plt.plot(yp, Pot_y_780/kb*1e3, label = "780nm")
plt.plot(yp, Pot_y_1057/kb*1e3, label = "1057nm")
plt.plot(yp, Pot_y/kb*1e3, label = "total")#,interpolation='none') 
plt.legend()
plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/squared_beam_potential_y.png')

plt.close()
