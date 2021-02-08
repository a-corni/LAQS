import numpy as np
np.random.seed()

import multiprocessing as mp

from MC_ivp_2 import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar
import time



def simulation_npart(nb, Intensities):
    
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
    
    dt = 0.0025*tend2 # 0.25% of simulation time
    print("tend1 (in s) :", tend1*tL)
    print("tend2, dt (in s): ", tend2*tL, dt*tL)
    print("Maximum number of scattering for 1 atom during the simulation", gamma_scatter_max*tend2*tL/2/np.pi)
    print("Maximum number of scattering for 1 atom during one step", gamma_scatter_max*dt*tL/2/np.pi)
    
    #initial speeds in vL units #Maxwell-Botzmann distribution
    vzr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    vyr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    vxr = np.random.normal(loc=0.0,scale=np.sqrt(kb*Ti/mass),size=nb)/vL
    # v/vL = sqrt(kbTi/m)/sqrt(2kbTp/m) = sqrt(Ti/(2Tp))
    
    #Definition of the gaussian beam
    global Ilight_780 = Intensities[0] 
    global I_1057 = Intensities[1]
    global Cnorm_1057 = Intensities[2]
    global grid_spacing = Intensities[3]
    
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
        kz = 1/1057/1e-9*2.0*np.pi #in rad.m-1
        
        Exs = Ex_1057*np.exp(-1j*kz*pz)+Ex_1057*np.exp(1j*kz*z) #um.kg.s-3.A-1 
        Eys = Ey_1057*np.exp(-1j*kz*pz)+Ey_1057*np.exp(1j*kz*z)
        Ezs = Ez_1057*np.exp(-1j*kz*pz)-Ez_1057*np.exp(1j*kz*z)  #pi shift
        
        Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
        Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
        Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 
        
        Its = Ixs+Iys+Izs
        Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 

        impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
        Islight_1057 = Is_1057/(2.0*impedance) #W.m-2
 
        #-------------------------------
        #compute the potential of the dipole trap
        #-------------------------------
        gamma = 2*np.pi*5.234e6 #Hz #decay rate
        delta_1057 = 2*np.pi/1057*1e9*cst.c-wa
        delta_780 = 2*np.pi/780*1e9*cst.c-wa
        Nanofiber_potential = 3*np.pi*cst.c**2/2/wa**3*gamma*(1/delta_1057*Islight_1057+1/delta_780*Ilight_780)
        
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
        
        #-----------------------------------
        #compute the gradient of the potential
        #get the dipole force
        #-----------------------------------
        def grad_field(I):
            size = np.shape(I)
            grad_I_x = np.zeros(size)
            grad_I_x[0,:] = (I[1,:]-I[-1,:])/(2*grid_spacing)
            grad_I_x[size[0]-1,:] = (I[0,:]-I[-2,:])/(2*grid_spacing)
            for i in range(1,size[0]-1):
                grad_I_x[i,:] = (I[i+1,:]-I[i-1,:])/(2*grid_spacing)
                
            grad_I_y = np.zeros(size)
            grad_I_y[:,0] = (I[:,1]-I[:,-1])/(2*grid_spacing)
            grad_I_y[:,size[1]-1] = (I[:,0]-I[:,-2])/(2*grid_spacing)
            for i in range(1,size[1]-1):
                grad_I_y[:,i] = (I[:,i+1]-I[:,i-1])/(2*grid_spacing)    
            
            return np.array([grad_I_x,grad_I_y])
        
        #grad_It_780 = grad_field(Ilight_780)
        #grad_It_1057 = grad_field(Islight_1057)
        grad_U0 = grad_field(Nanofiber_potential)
        
        return [-g*tL/vL, vz, -1/mass*tL/vL*interpolate(grad_U0[1], py*L, px*L), vy, -1/mass*tL/vL*interpolate(grad_U0[0], py*L, px*L), vx]
    
    def fun_scattering(t,y):
        pz = y[1]
        vz = y[0]
        
        py = y[3]
        vy = y[2]
        
        px = y[5]
        vx = y[4]
        
        mu = 2*np.random.uniform(0,1)-1 #cos(theta)
        phi = 2*np.random.uniform(0,1)*np.pi
        vz += hbar*k/mass*(1 + np.sqrt(1-mu**2)*np.cos(phi))
        vy += hbar*k/mass*np.sqrt(1-mu**2)*np.sin(phi)
        vx += hbar*k/mass*mu
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
        
        Exs = Ex_1057*np.exp(-1j*kz*pz)+Ex_1057*np.exp(1j*kz*z) #um.kg.s-3.A-1 
        Eys = Ey_1057*np.exp(-1j*kz*pz)+Ey_1057*np.exp(1j*kz*z)
        Ezs = Ez_1057*np.exp(-1j*kz*pz)-Ez_1057*np.exp(1j*kz*z)  #pi shift
        
        Ixs = np.array([np.real(E*np.conj(E)) for E in Exs]) #um².kg².s-6.A-2 
        Iys = np.array([np.real(E*np.conj(E)) for E in Eys]) 
        Izs = np.array([np.real(E*np.conj(E)) for E in Ezs]) 
        
        Its = Ixs+Iys+Izs
        Is_1057 = Its*Cnorm_1057 # where Cnorm thus normalizes the e-field of one beam.. 

        impedance = np.sqrt(cst.mu_0/cst.epsilon_0) 
        Islight_1057 = Is_1057/(2.0*impedance) #W.m-2
 
        #-------------------------------
        #compute the potential of the dipole trap
        #-------------------------------
        gamma = 2*np.pi*5.234e6 #Hz #decay rate
        delta_1057 = 2*np.pi/1057*1e9*cst.c-wa
        delta_780 = 2*np.pi/780*1e9*cst.c-wa
        Nanofiber_potential = 3*np.pi*cst.c**2/2/wa**3*gamma*(1/delta_1057*Islight_1057+1/delta_780*Ilight_780)
        
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
        
        return gamma_scatter_0*interpolate(Nanofiber_potential, y[3], y[5])*dt
    
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
        
        thetai = np.random.uniform(0, np.pi)
        phii = np.random.uniform(0, 2*np.pi)
        ri = np.random.uniform(0,1e-3)/L
        zi = ri*np.cos(thetai)
        yi = ri*np.sin(thetai)*np.sin(phii)
        xi = ri*np.sin(thetai)*np.cos(phii)
    
        #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
        
        sol = solve_ivp(af1, fun_scattering, proba_scattering, (tstart1,tend1), (vzi,zi,vyi,yi,vxi,xi), method = BESTMETHOD, t_eval = np.linspace(tstart1,tend1,int((tend1-tstart1)/dt),endpoint=True),atol = 1e-10, rtol = 1e-6, dense_output = False, events = None, vectorized = False)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
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
    
        if (yri*L)**2+(xri*L)**2<w0**2/2:
            if 1/2*mass*((vyri*vL)**2+(vxri*vL)**2) < Umax:
                zr.append(zri)
                yr.append(yri)
                xr.append(xri)
                vzr.append(vzri)
                vyr.append(vyri)
                vxr.append(vxri)
    
    def af2(t,y):
        
        pz = y[1]
        vz = y[0]
        
        py = y[3]
        vy = y[2]
        
        px = y[5]
        vx = y[4]
        
        # red detuned
        return [-g*tL/vL-U0/mass*tL/vL*derivTz(pz*L,py*L,px*L), vz, -U0/mass*tL/vL*derivTy(pz*L,py*L,px*L), vy, -U0/mass*tL/vL*derivTx(pz*L,py*L,px*L), vx]
    
    start2 = time.time()
    BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
    #BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method
    
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
        
        sol = solve_ivp(af2, fun_scattering, proba_scattering, (tstart2,tend2), (vzi,zi,vyi,yi,vxi,xi), method=BESTMETHOD, t_eval = np.linspace(tstart2,tend2,int((tend2-tstart2)/dt),endpoint=True), atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
        sol_arr_2.append(sol)
        
    end2 = time.time()
    pbar.finish()
    print('Part 1 finished in (seconds):', end2 - start2)
    
    return sol_arr_2