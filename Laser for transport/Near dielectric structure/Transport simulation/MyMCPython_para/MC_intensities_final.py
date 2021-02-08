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
    
    filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-epsilon.780.h5'  
    data = h5py.File(filename,'r') 
    
    eps = data['data'][:,:] 
    rindex = np.sqrt(eps) # get the refractive index from relative permittivity epsilon_r = n^2 
    grid_spacing = SX/eps.shape[0] #um
    resolution = eps.shape[0]/8  #nbpixels/um
    
    #-------------------------------------
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
    #
    #careful I got it wrong in terms of computing the surface element value.. surface/(nbgridpoint1x*nbgridpoint1y) which is not 128 but 256.. 128 is the resolution 
    #power P => integrate over entire transverse plane 
    surface_element = (grid_spacing)**2#grid_spacing**2 #um**2
    #
    P_780 = np.sum(Sz_780)*surface_element #kg.um².s-3 
    print('Power in kg.(um)².s-3', P_780) 
    
    #load total energy density 
    filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-tot.rpwr.k01.b01.yodd.780.h5'  
    data_780_e = h5py.File(filename,'r') 
    #
    utot_780 = 2.0/SX*data_780_e['data'][:,:] #kg.um².s-3/um²# cycle-average total EM energy density  
    #
    #integrate over cell (total energy) 
    Utot_780 = np.sum(utot_780)*surface_element #kg.um².s-3
    
    vg_780 = P_780/Utot_780 #unitary 
    
    #in SI units now 
    vg_780 = vg_780*cst.c #m.s-1
    
    #-------------------------------------
    # normalize fields
    
    P_780 = vg_780*cst.epsilon_0*0.5*np.sum(np.multiply(It_780,eps))*(SX*1.0e-6*SY*1.0e-6)/(It_780.shape[0]*It_780.shape[1]) #kg.m².s-3
    
    Power_Wanted_780 = 12e-3 #Watt # nanofiber exp = 8 mW
    Cnorm_780 = Power_Wanted_780/P_780 
    
    I_780 = It_780*Cnorm_780 #W.m-2
    
    #-------------------------------------
    #load E-field data for 1057 nm light
    #-------------------------------------
    
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
    
    #load total energy density 
    filename = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber-tot.rpwr.k01.b01.yeven.1057.h5'  
    data_1057_e = h5py.File(filename,'r') 
    utot_1057 = 2.0/SX*data_1057_e['data'][:,:] # cycle-average total EM energy density 
    
    #integrate over cell (total energy) 
    Utot_1057 = np.sum(utot_1057)*surface_element 
    
    vg_1057 = P_1057/Utot_1057 
    
    #in SI units now 
    vg_1057 = vg_1057*cst.c 
    
    #-------------------------------------
    # normalize field
    
    P_1057 = vg_1057*cst.epsilon_0*0.5*np.sum(np.multiply(It_1057,eps))*(SX*1.0e-6*SY*1.0e-6)/(It_1057.shape[0]*It_1057.shape[1]) 
    
    Power_Wanted_1057 = 1.1e-3 #Watt  nanofiber experiment = 1.5 mW
    Cnorm_1057 = Power_Wanted_1057/P_1057  
    
    #I_1057 = It_1057*Cnorm_1057
    
    return (Ex_top, Ey_top, Ez_top, Ex_bottom, Ey_bottom, Ez_bottom, Cnorm_top, I_780, Ex_1057, Ey_1057, Ez_1057, Cnorm_1057, grid_spacing) 