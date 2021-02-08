import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors
plt.close('all')
from scipy import constants as cst
import cesium_data as cs 
#import cesium_data as cs

# plot 2D projection of the atoms
# back of all images

#3D trajectories
simulation_name = "Nanofiber_conveyor_belt"
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/solverp0_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']

nb_part = sol_arr.shape[0]
nb_iter = len(sol_arr[0].t)

SX = 14.0
SY = SX
lam = 980e-9
k = 2*np.pi/lam
f = cst.c/lam

print("frequency : ", f)

T = 1/f
w = 2*np.pi*f

 
df = 1e6
w1 = w

tend = 10/df
dt = 0.01*tend # 0.25% of simulation time
print("simulation duration : ", tend)
time = [i/nb_iter*tend for i in range(nb_iter)]

def deltaf(t) :
    
    df = 1e6
    return 2*np.pi*df*t 

# set the resolution higher to 512 to check that beta is closer to analytic solution 

#print('beta:', 1.342308553122232*2.0*np.pi/1.0e-6) 
#1.342308553122232 

#-------------------------------------
# #load relative electric permittivity  
filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/eps-000000000.h5'  
data = h5py.File(filename,'r') 
eps = data['eps'][:,:] 
rindex = np.sqrt(eps) # get the refractive index from relative permittivity epsilon_r = n^2 
print(eps.shape)
grid_spacing = SX/eps.shape[1] #um
print('grid spacing (um)', grid_spacing)
resolution = eps.shape[1]/SX  #nbpixels/um
print('resolution', resolution)

#-------------------------------------
#load E-field data for 780 nm light
filename_top = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_meep-out/e-000005120.h5'  
data_top = h5py.File(filename_top,'r') 
filename_bottom = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_meep2-out/e-000005120.h5'  
data_bottom = h5py.File(filename_bottom,'r') 
#keys = list(data.keys()) 
#print(keys) 
Ex_top = np.array(data_top['ex.r'][:,:]) + 1j*np.array(data_top['ex.i'][:,:]) #um.kg.s-3.A-1 
Ey_top = np.array(data_top['ey.r'][:,:]) + 1j*np.array(data_top['ey.i'][:,:])
Ez_top = np.array(data_top['ez.r'][:,:]) + 1j*np.array(data_top['ez.i'][:,:])

Ex_bottom = np.array(data_bottom['ex.r'][:,:]) + 1j*np.array(data_bottom['ex.i'][:,:]) #um.kg.s-3.A-1 
Ey_bottom = np.array(data_bottom['ey.r'][:,:]) + 1j*np.array(data_bottom['ey.i'][:,:]) 
Ez_bottom = np.array(data_bottom['ez.r'][:,:]) + 1j*np.array(data_bottom['ez.i'][:,:])
print(Ex_bottom[len(Ex_bottom)//2,len(Ex_bottom)//2],Ey_bottom[len(Ey_bottom)//2-100,len(Ey_bottom)//2-100],Ez_bottom[0,0])
print(Ex_top[len(Ex_top)//2,len(Ex_top)//2],Ey_bottom[len(Ey_bottom)//2-100,len(Ey_bottom)//2+100],Ez_bottom[0,0])

Ix_top = np.array([np.real(E*np.conj(E)) for E in Ex_top]) #um².kg².s-6.A-2 
Iy_top = np.array([np.real(E*np.conj(E)) for E in Ey_top]) 
Iz_top = np.array([np.real(E*np.conj(E)) for E in Ez_top]) 

Ix_bottom = np.array([np.real(E*np.conj(E)) for E in Ex_bottom]) #um².kg².s-6.A-2 
Iy_bottom = np.array([np.real(E*np.conj(E)) for E in Ey_bottom]) 
Iz_bottom = np.array([np.real(E*np.conj(E)) for E in Ez_bottom]) 

#total electric field intensity (NOT light intensity)
It_top = Ix_top + Iy_top + Iz_top
It_bottom = Ix_bottom + Iy_bottom + Iz_bottom
impedance = np.sqrt(cst.mu_0/cst.epsilon_0)
Ilight_top = It_top/(2.0*impedance)
P_top = np.pi*(1e-6)**2/2*np.max(Ilight_top)

Ilight_bottom = It_bottom/(2.0*impedance)
P_bottom = np.pi*(1e-6)**2/2*np.max(Ilight_bottom)

Power_Wanted = 14e-3 #Watt # nanofiber exp = 8 mW
Cnorm_top = Power_Wanted/P_top
Cnorm_bottom = Power_Wanted/P_bottom

xp = np.linspace(-SX/2, SX/2, It_top.shape[0], endpoint=False) 
yp = np.linspace(-SY/2, SY/2, It_top.shape[1], endpoint=False) 

def potential(fieldint,wavelength):
    #fieldint is the field intensity |E|^2.. no impedance involved etc..
    return -1.0/4.0*cs.alphaComplet(wavelength)*fieldint
    
kb = 1.3806504e-23 # J/K # Boltzmann constant

for titer in range(0, nb_iter, 3):
    
    plt.figure(titer)
    
    t = time[titer]
            
    Ex = Ex_top + Ex_bottom*np.exp(-1j*deltaf(t))
    Ey = Ey_top + Ey_bottom*np.exp(-1j*deltaf(t))
    Ez = Ez_top + Ez_bottom*np.exp(-1j*deltaf(t))
    
    Ix = Cnorm_top*np.array([np.real(E*np.conj(E)) for E in Ex]) #um².kg².s-6.A-2 
    Iy = Cnorm_top*np.array([np.real(E*np.conj(E)) for E in Ey]) 
    Iz = Cnorm_top*np.array([np.real(E*np.conj(E)) for E in Ez]) 
    It = Ix+Iy+Iz
    
    Nanofiber_potential = potential(It, 2*np.pi*cst.c/w1)
    r0 = 0.150
    
    for i,x in enumerate(xp):
        for j,y in enumerate(yp):
            
            r = np.sqrt(xp[i]**2+yp[j]**2)
            
            if r < r0:
                Nanofiber_potential[i,j] = 0
            else:
                Nanofiber_potential[i,j] -=  (5.6E-49)/(((r-r0)*1e-6)**3) #add vanderwals potential
    px = [sol_arr[j].y[3,titer]*1e6 for j in range(nb_part)]
    py = [sol_arr[j].y[1,titer]*1e6 for j in range(nb_part)]
    
    #plot 2D image of the potential
    plt.xlabel('y (um)') 
    plt.ylabel('x (um)') 
    plt.xlim([-SX/2, SX/2]) 
    plt.ylim([-SY/2, SY/2]) 
    
    #levels = [-2.5,-2,-1.5,-1,-0.5,0,1,2,3,4,5]
    plt.title('Simulation t = '+ '%2.f'%(t*1e6)+ ' us') 
    plt.imshow(Nanofiber_potential/kb*1e3, norm = colors.DivergingNorm(vmin=-10, vcenter= -0.3, vmax=1.0), cmap=plt.cm.RdYlBu,extent=[xp[0],xp[-1],yp[0],yp[-1]])#,interpolation='none') 
    cbar = plt.colorbar() 
    cbar.set_label('Trap Potential [in mK]',rotation=270,labelpad=20)
    plt.tight_layout()
    plt.scatter(py, px, s = 0.1)
    plt.savefig('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/Nanofiber_conv_belt'+str(titer)+'.png')
    