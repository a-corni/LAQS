import numpy as np
import multiprocessing as mp

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
#from scipy.special import eval_genlaguerre
#from scipy.special import genlaguerre
from scipy import constants as cst 
import cesium_data as cs
import progressbar
import time

def simulation_npart(nb, Intensities):
    
    np.random.seed(int(1e6*time.time())%2**32)

    #Fundamental Physical Constants
    kb = 1.3806504e-23 # J/K # Boltzmann constant
    g = 9.81 #m.s-2
    
    #Cesium Physical Properties
    mass = 2.20694657e-25 #kg #cesium mass
    
    #Cesium D2 Transition Optical Properties
    wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
    lambda_a = 852.34727582e-9 #m #wavelength
    gamma = 2*np.pi*5.234e6 #Hz #decay rate
    Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)
    
    # Length, speed, time constants
    vL = 1     #= np.sqrt(2*U0/mass)
    tL = 1     #= np.sqrt(mass/U0)*w0/2
    L = 1      #= w0/np.sqrt(2)
    print("L, vL, tL", L, vL, tL)
    
    #Cesium cloud parameters
    Ti = 30e-6 #K #Initial temperature of the cloud #Doppler Temperature
    
    #Simulation's time duration and step
    #Definition of the simulation's duration
    tend2 = 10e-5
        
    #First part of the simulation : dispersion of the cloud
    
    tstart1 = 0.0
    tend1 = 0.01*tend2 #1% of simulation time
    tstart2 = tend1
    
    #Simulation's time step
    
    dt = 0.0025*tend2 # 0.25% of simulation time
    print("tend1 (in s) :", tend1*tL)
    print("tend2, dt (in s): ", tend2*tL, dt*tL)
    
    #initial speeds in vL units #Maxwell-Botzmann distribution
    vzr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    vyr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    vxr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    # v/vL = sqrt(kbTi/m)/sqrt(2kbTp/m) = sqrt(Ti/(2Tp))
    
    #Definition of the gaussian beam
    
    I_780 = Intensities[0] 
    Ex_1057 = Intensities[1]
    Ey_1057 = Intensities[2]
    Ez_1057 = Intensities[3]
    Cnorm_1057 = Intensities[4]
    grid_spacing = Intensities[5]
    
    def af1(t,y):
        
        pz = y[1]
        vz = y[0]
        
        py = y[3]
        vy = y[2]
        
        px = y[5]
        vx = y[4]
        
        #-------------------------------------
        # now we compute a standing-wave red fields (imagine retro-reflection on a mirror)
        # at a given z (we skip the time phase omega*t)
        kz = 1.013197483552876*2.0*np.pi/1.0e-6 #in rad.m-1
        
        # Exs = Ex_1057*np.exp(-1j*kz*pz)+Ex_1057*np.exp(1j*kz*pz) #um.kg.s-3.A-1 
        # Eys = Ey_1057*np.exp(-1j*kz*pz)+Ey_1057*np.exp(1j*kz*pz)
        # Ezs = Ez_1057*np.exp(-1j*kz*pz)-Ez_1057*np.exp(1j*kz*pz)  #pi shift
        # 
        # Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
        # Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
        # Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 
        # 
        # Its = Ixs + Iys + Izs
        # Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 
 
        #-------------------------------
        #compute the potential of the dipole trap
        #-------------------------------
        
        def potential(fieldint,wavelength):
            #fieldint is the field intensity |E|^2.. no impedance involved etc..
            return -1.0/4.0*cs.alphaComplet(wavelength)*fieldint
        
        # Nanofiber_potential = potential(Is_1057, 1057.0e-9) +potential(I_780, 780.0e-9)
        
        def close_points(field, x, y):
            
            #provide y, x in m
            #bilinear interpolation
            #SX = 8
            ystar = y + 4*1e-6 #m
            xstar = x + 4*1e-6 #m
            ix = xstar/grid_spacing/1e-6 #index
            iy = ystar/grid_spacing/1e-6 #index
            x1 = int(ix)
            y1 = int(iy)
            dxstar = ix-x1
            dystar = iy-y1
            
            (field11, field21, field12, field22) = (field[y1%field.shape[1],x1%field.shape[0]],field[y1%field.shape[1], (x1+1)%field.shape[0]], field[(y1+1)%field.shape[1],x1%field.shape[0]], field[(y1+1)%field.shape[1],(x1+1)%field.shape[0]])
            (field01, field02, field10, field20, field13, field23, field31, field32 ) = (field[y1%field.shape[1],(x1-1)%field.shape[0]],field[(y1+1)%field.shape[1],(x1-1)%field.shape[0]], field[(y1-1)%field.shape[1], x1%field.shape[0]], field[(y1-1)%field.shape[1],(x1+1)%field.shape[0]], field[(y1+2)%field.shape[1],x1%field.shape[0]], field[(y1+2)%field.shape[1],(x1+1)%field.shape[0]], field[y1%field.shape[1],(x1+2)%field.shape[0]], field[(y1+1)%field.shape[1],(x1+2)%field.shape[0]])
            
            return (np.array([field11, field21, field12, field22, field01, field02, field10, field20, field13, field23, field31, field32]), dxstar, dystar)
        
        (Ex_1057_points, dxstar, dystar) = close_points(Ex_1057, px, py) 
        (Ey_1057_points, dxstar, dystar) = close_points(Ey_1057, px, py) 
        (Ez_1057_points, dxstar, dystar) = close_points(Ez_1057, px, py) 
        (I_780_points, dxstar, dystar) = close_points(I_780, px, py) 
        
        Exs_1057_points = np.array([Ex_1057_points*np.exp(-1j*kz*(pz-grid_spacing*1e-6)) + Ex_1057_points*np.exp(1j*kz*(pz-grid_spacing*1e-6)), Ex_1057_points*np.exp(-1j*kz*pz) + Ex_1057_points*np.exp(1j*kz*pz), Ex_1057_points*np.exp(-1j*kz*(pz+grid_spacing*1e-6)) + Ex_1057_points*np.exp(1j*kz*(pz+grid_spacing*1e-6))])
        Eys_1057_points = np.array([Ey_1057_points*np.exp(-1j*kz*(pz-grid_spacing*1e-6)) + Ey_1057_points*np.exp(1j*kz*(pz-grid_spacing*1e-6)),Ey_1057_points*np.exp(-1j*kz*pz) + Ey_1057_points*np.exp(1j*kz*pz),Ey_1057_points*np.exp(-1j*kz*(pz+grid_spacing*1e-6)) + Ey_1057_points*np.exp(1j*kz*(pz+grid_spacing*1e-6))])
        Ezs_1057_points = np.array([Ez_1057_points*np.exp(-1j*kz*(pz-grid_spacing*1e-6)) - Ez_1057_points*np.exp(1j*kz*(pz-grid_spacing*1e-6)),Ez_1057_points*np.exp(-1j*kz*pz) - Ez_1057_points*np.exp(1j*kz*pz),Ez_1057_points*np.exp(-1j*kz*(pz+grid_spacing*1e-6)) - Ez_1057_points*np.exp(1j*kz*(pz+grid_spacing*1e-6))])

        Ixs_points = np.array([np.real(E*np.conj(E)) for E in Exs_1057_points]) #um².kg².s-6.A-2 
        Iys_points = np.array([np.real(E*np.conj(E)) for E in Eys_1057_points]) 
        Izs_points = np.array([np.real(E*np.conj(E)) for E in Ezs_1057_points]) 
        
        Its_points = Ixs_points + Iys_points + Izs_points
        Is_1057_points = Its_points*Cnorm_1057        
        
        Nanofiber_potential = potential(Is_1057_points, 1057.0e-9) + potential(I_780_points, 780.0e-9)  
          
        def interpolate(field, dxstar, dystar):
            
            #bilinear interpolation
            
            field1 = field[0]
            field2 = field[1]
            field3 = field[2]
            field4 = field[3]
             
            return (field2 - field1)*dxstar + (field3-field1)*dystar + (field1+field4-field2-field3)*dxstar*dystar + field1
        
        #-----------------------------------
        #compute the gradient of the potential
        #get the dipole force
        #-----------------------------------
        def grad_field(I):
            
            grad_I_x = np.array([(I[1,1]-I[1,4])/(2*grid_spacing), (I[1,10]-I[1,0])/(2*grid_spacing), (I[1,3]-I[1,5])/(2*grid_spacing),(I[1,11]-I[1,1])/(2*grid_spacing)]) 
            
            grad_I_y = np.array([(I[1,2]-I[1,6])/(2*grid_spacing), (I[1,3]-I[1,7])/(2*grid_spacing), (I[1,8]-I[1,0])/(2*grid_spacing),(I[1,9]-I[1,1])/(2*grid_spacing)])

            grad_I_z = np.array([(I[2,0]-I[0,0])/(2*grid_spacing), (I[2,1]-I[0,1])/(2*grid_spacing), (I[2,2]-I[0,2])/(2*grid_spacing),(I[2,3]-I[0,3])/(2*grid_spacing)])
            
            return np.array([grad_I_x*1e6,grad_I_y*1e6, grad_I_z*1e6])
        
        #grad_It_780 = grad_field(Ilight_780)
        #grad_It_1057 = grad_field(Islight_1057)
        grad_U = grad_field(Nanofiber_potential)
        
        g = 9.81 #m.s-2
        r0 = 0.235*1e-6
        r = np.sqrt(px**2+py**2)*L
        
        return [ -1/mass*tL/vL*interpolate(grad_U[2], dxstar*L, dystar*L), vz, -g*tL/vL -1/mass*tL/vL*interpolate(grad_U[1], dxstar*L, dystar*L) + 3*5.6e-49*py/r/(r-r0)**4, vy, -1/mass*tL/vL*interpolate(grad_U[0], dxstar*L, dystar*L) + 3*5.6e-49*py/r/(r-r0)**4, vx]
    
    def fun_scattering(t,y):
        
        kz = 1/1057/1e-9*2.0*np.pi
        pz = y[1]
        vz = y[0]
        
        py = y[3]
        vy = y[2]
        
        px = y[5]
        vx = y[4]
        
        mu = 2*np.random.uniform(0,1)-1 #cos(theta)
        phi = 2*np.random.uniform(0,1)*np.pi
        vz += cst.hbar*kz/mass*(1 + np.sqrt(1-mu**2)*np.cos(phi))
        vy += cst.hbar*kz/mass*np.sqrt(1-mu**2)*np.sin(phi)
        vx += cst.hbar*kz/mass*mu
        return [vz, pz, vy, py, vx, px]
    
    def proba_scattering(dt, y):
        
        pz = y[1]
        vz = y[0]
        
        py = y[3]
        vy = y[2]
        
        px = y[5]
        vx = y[4]
        
        #-------------------------------------
        # now we compute a standing-wave red fields (imagine retro-reflection on a mirror)
        # at a given z (we skip the time phase omega*t)
        kz = 1/1057/1e-9*2.0*np.pi #in rad.m-1
        
        Exs = Ex_1057*np.exp(-1j*kz*pz)+Ex_1057*np.exp(1j*kz*pz) #um.kg.s-3.A-1 
        Eys = Ey_1057*np.exp(-1j*kz*pz)+Ey_1057*np.exp(1j*kz*pz)
        Ezs = Ez_1057*np.exp(-1j*kz*pz)-Ez_1057*np.exp(1j*kz*pz)  #pi shift
        
        Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
        Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
        Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 
        
        Its = Ixs+Iys+Izs
        Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 
 
        #-------------------------------
        #compute the potential of the dipole trap
        #-------------------------------
        def potential(fieldint,wavelength):
            #fieldint is the field intensity |E|^2.. no impedance involved etc..
            return -1.0/4.0*cs.alphaComplet(wavelength)*fieldint
        
        Nanofiber_potential = potential(Is_1057, 1057.0e-9) +potential(I_780,780.0e-9)

        def interpolate(field, y, x):
            
            ystar = y+4*1e-6
            xstar = x+s*1e-6
            ix = xstar/grid_spacing
            iy = ystar/grid_spacing
            x1 = int(ix)
            y1 = int(iy)
            dxstar = (xstar-x1)/grid_spacing
            dystar = (ystart-y1)/grid_spacing
            (field1, field2, field3, field4) = (field[x1,y1],field[x1+grid_spacing, y1], field[x1, y1+grid_spacing], field[x1+grid_spacing, y1+grid_spacing])
             
            return field1*np.sqrt(dxstar**2+dystar**2) + field2*np.sqrt((1-dxstar)**2+dystar**2)+field3*np.sqrt((1-dystar)**2+dxstar**2)+field4*((1-dxstar)**2+(1-dystar)**2)
        
        r0 = 0.235*1e-6
        r = np.sqrt(px**2+py**2)*L
        
        return (interpolate(Nanofiber_potential, y[3], y[5])-(5.6E-49)/((r-r0)**3))*dt
    
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
        
        thetai = np.random.uniform(0, 2*np.pi)
        ri = np.random.uniform(0,0.5*1e-6)/L
        zi = 0
        yi = ri*np.sin(thetai)
        xi = 0.6*1e-6+ri*np.cos(thetai)
    
        #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
        
        sol = solve_ivp(af1, (tstart1,tend1), (vzi,zi,vyi,yi,vxi,xi), method = BESTMETHOD, t_eval = np.linspace(tstart1, tend1, int((tend1-tstart1)/dt),endpoint=True),atol = 1e-10, rtol = 1e-6, dense_output = False, events = None, vectorized = False)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
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
    
        if (yri*L)**2+(xri*L)**2<1e-6**2:
            if (yri*L)**2+(xri*L)**2>0.235*1e-6**2:
                zr.append(zri)
                yr.append(yri)
                xr.append(xri)
                vzr.append(vzri)
                vyr.append(vyri)
                vxr.append(vxri)
    
    start2 = time.time()
    BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
    #BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stiff) and nonstiff Adams'method
    
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
        
        sol = solve_ivp(af1, (tstart2,tend2), (vzi,zi,vyi,yi,vxi,xi), method=BESTMETHOD, t_eval = np.linspace(tstart2,tend2,int((tend2-tstart2)/dt),endpoint=True), atol=1e-6,rtol=1e-10) #,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
        sol_arr_2.append(sol)
        
    end2 = time.time()
    pbar.finish()
    print('Part 1 finished in (seconds):', end2 - start2)
    
    return sol_arr_2