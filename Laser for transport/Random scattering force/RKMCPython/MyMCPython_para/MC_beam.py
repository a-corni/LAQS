import numpy as np
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
#simulation to run
import numpy as np
np.random.seed()
import MC_physics
import MC_Particles

class Beam
    #Beam Properties
    delta = 10e9 #Hz
    wl = wa + delta #Hz
    k = wl/c #m-1
    la = 2.0*np.pi/k #m # wave-vector
    w0 = 1e-3 #m # waist 1/e^2 intensity radius
    zR = np.pi*w0**2/la #m # Rayleigh range
    z0 = 0.0 #m #position of maximum intensity along the z-axis
    
    #Doughnut beam : described using Laguerre polynomials of parameter l=1, p=0. 
    l = 1 #azimuthal index >-1
    p = 0 #radial index (positive)
    
    # Potential depth in Joules, U0 = kB*T
    P0 = 100e-3 #W #Laser power
    I0 = 2*P0/np.pi/w0**2 #W/m**2 #Laser intensity
    Imax = I0*np.exp(-1)
    U0 = 3*np.pi*c**2/2/wa**3*gamma/delta*I0
    Umax = 3*np.pi*c**2/2/wa**3*gamma/delta*Imax #J #Potential depth
    Tp = Umax/kb #°K #Trapping potential in Temperature 
    
    # Scattering rate
    gamma_scatter_0 = U0*gamma/delta/hbar
    gamma_scatter_max = Umax*gamma/delta/hbar
    
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