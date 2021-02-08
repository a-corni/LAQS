import h5py 
import numpy as np 
import matplotlib.pyplot as plt 
plt.close('all')
from scipy import constants as cst 
#import cesium_data as cs

SX = 8.0
SY = SX
# set the resolution higher to 512 to check that beta is closer to analytic solution 

#print('beta:', 1.342308553122232*2.0*np.pi/1.0e-6) 
#1.342308553122232 

#-------------------------------------
# #load relative electric permittivity
def get_Intensities():  
        
    filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MyMCPython_para/eps-000000000.h5'  
    data = h5py.File(filename,'r') 
    eps = data['eps'][:,:] 
    rindex = np.sqrt(eps) # get the refractive index from relative permittivity epsilon_r = n^2 
    grid_spacing = SX/eps.shape[0] #um
    resolution = eps.shape[0]/SX  #nbpixels/um
    
    #-------------------------------------    
    filename_top = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MyMCPython_para/e-000005120.h5'  
    data_top = h5py.File(filename_top,'r') 
    filename_bottom = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MyMCPython_para/e-000005120-2.h5'  
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
    
    return (Ex_top, Ey_top, Ez_top, Ex_bottom, Ey_bottom, Ez_bottom, Cnorm_top) 