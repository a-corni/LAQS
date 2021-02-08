from sympy import *
import numpy as np
np.random.seed()

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre
from scipy.misc import derivative

import progressbar
import time

g = 9.81
c = 2.998*10**8 #m.s-1
wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

#Beam Properties
delta1 = -1e9 #Hz
wl1 = wa + delta1 #Hz
k1 = wl1/c #m-1
la1 = 2.0*np.pi/k1 #m # wave-vector
w0 = 1e-3 #m # waist 1/e^2 intensity radius
zR1 = np.pi*w0**2/la1 #m # Rayleigh range
print(zR1)
z0 = 0.

delta2 = -2e9 #Hz
wl2 = wa + delta2 #Hz
k2 = wl2/c #m-1
la2 = 2.0*np.pi/k2 #m # wave-vector
w0 = 1e-3 #m # waist 1/e^2 intensity radius
zR2 = np.pi*w0**2/la2 #m # Rayleigh range
print(zR2)
z0 = 0.
# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) #?
tL = L/vL

#trap freq
wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(2*U0/(mass*zR1**2))/(2*np.pi)

tstart = 0.0
nper = 20 # nb oscill periods
tend_sim = 5*nper*(1.0/wx)/tL/2 # !!! time is in unit of tL
tend = tend_sim*tL
dt = 0.05*(1.0/wx)/tL # 5% period in tL unit

print(wx,wz)

print("L,vL,tL",L,vL,tL)

l=0
p=0

#Definition of the gaussian beam
def pos(t):
    D = 0.5
    if t<=tend/4:
        return D*(-7040/180*(t/tend)**5+320/12*(t/tend)**4)
    elif t<=3*tend/4 :
        return D*(3200/180*(t/tend)**5-1600/36*(t/tend)**4+640/18*(t/tend)**3-160/18*(t/tend)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*t/tend)+D*(-7040/180*(1/4)**5+320/12*(1/4)**4)-D*(3200/180*(1/4)**5-1600/36*(1/4)**4+640/18*(1/4)**3-160/18*(1/4)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*1/4)
    else :
        return D*(-7040/180*(t/tend)**5+6080/36*(t/tend)**4-5120/18*(t/tend)**3+4160/18*(t/tend)**2-(-7040/36+6080/9-5120/6+4160/9)*t/tend)-D*(-7040/180+6080/36-5120/18+4160/18-(-7040/36+6080/9-5120/6+4160/9))+D


#Waist at z
def w1(z):
    return w0*np.sqrt(1.0+((z-z0)/zR1)**2)
def w2(z):
    return w0*np.sqrt(1.0+((z-z0)/zR2)**2)
#Intensity of the beam    
def It(z,y,x,t):
    #Unitary intensity
    #Amplitude added in the force expression
    r_w_z_2_1 = (y**2+x**2)/w1(z)**2
    r_w_z_2_2 = ((z+pos(t))**2+y**2)/w2(x)**2

    return (w0/w1(z))**2*(2*r_w_z_2_1)**l*np.exp(-2*r_w_z_2_1)*eval_genlaguerre(p,l,2*r_w_z_2_1)**2+ (w0/w2(x))**2*(2*r_w_z_2_2)**l*np.exp(-2*r_w_z_2_2)*eval_genlaguerre(p,l,2*r_w_z_2_2)**2
    
def derivTx(z,y,x,t):
    r_w_z_2_1 = (y**2+x**2)/w1(z)**2
    r_w_z_2_2 = ((z+pos(t))**2+y**2)/w2(x)**2
    lag_1 = eval_genlaguerre(p,l,2*r_w_z_2_1)
    lag_2 = eval_genlaguerre(p,l,2*r_w_z_2_2)

    return (w0/w1(z))**2*(2*r_w_z_2_1)**(l-1)*np.exp(-2*r_w_z_2_1)*lag_1*(4*x/w1(z)**2)*((l-2*r_w_z_2_1)*lag_1+4*r_w_z_2_1*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_1))+np.exp(-2*r_w_z_2_2)*lag_2*2*(2*r_w_z_2_2)**l*la2**2*x/np.pi**2/w2(x)**4*((-(1+l)+2*r_w_z_2_2)*lag_2-4*r_w_z_2_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_2))
    
def derivTy(z,y,x,t):
    r_w_z_2_1 = (y**2+x**2)/w1(z)**2
    r_w_z_2_2 = ((z+pos(t))**2+y**2)/w2(x)**2
    lag_1 = eval_genlaguerre(p,l,2*r_w_z_2_1)
    lag_2 = eval_genlaguerre(p,l,2*r_w_z_2_2)

    return (w0/w1(z))**2*(2*r_w_z_2_1)**(l-1)*np.exp(-2*r_w_z_2_1)*lag_1*(4*y/w1(z)**2)*((l-2*r_w_z_2_1)*lag_1+4*r_w_z_2_1*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_1))+(w0/w2(x))**2*(2*r_w_z_2_2)**(l-1)*np.exp(-2*r_w_z_2_2)*lag_2*(4*y/w2(x)**2)*((l-2*r_w_z_2_2)*lag_2+4*r_w_z_2_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_2))
    
def derivTz(z,y,x,t):
    r_w_z_2_1 = (y**2+x**2)/w1(z)**2
    r_w_z_2_2 = ((z+pos(t))**2+y**2)/w2(x)**2
    lag_1 = eval_genlaguerre(p,l,2*r_w_z_2_1)
    lag_2 = eval_genlaguerre(p,l,2*r_w_z_2_2)

    return np.exp(-2*r_w_z_2_1)*lag_1*2*(2*r_w_z_2_1)**l*la1**2*z/np.pi**2/w1(z)**4*((-(1+l)+2*r_w_z_2_1)*lag_1-4*r_w_z_2_1*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_1))+(w0/w2(x))**2*(2*r_w_z_2_2)**(l-1)*np.exp(-2*r_w_z_2_2)*lag_2*(4*(z+pos(t))/w2(x)**2)*((l-2*r_w_z_2_2)*lag_2+4*r_w_z_2_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2_2))

def af2(t,y):
    
    pz = y[1]
    vz = y[0]
    
    py = y[3]
    vy = y[2]
    
    px = y[5]
    vx = y[4]
    
    # red detuned
    return [-g*tL/vL+U0/mass*tL/vL*derivTz(pz*L,py*L,px*L,t*tL), vz, +U0/mass*tL/vL*derivTy(pz*L,py*L,px*L,t*tL), vy, +U0/mass*tL/vL*derivTx(pz*L,py*L,px*L,t*tL), vx]


start = time.time()
BESTMETHOD = 'RK45'# actually Dorman-Prince 5,  
#BESTMETHOD = 'LSODA'=>'LSODA' switches between BDF (for stikkf) and nonstiff Adams'method

print("start Dorman-Prince 5 algo:")

nb = 20 # nb particles
#vr = maxwell.rvs(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb) #for T = 0.1 milliKelvin
vzr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
vyr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
vxr = np.random.normal(loc=0.0,scale=np.sqrt(0.1/2.0),size=nb)
# scale a = sqrt(kT/m) => and v in units of vL = sqrt(2U0/mass) => vL^2/a^2 = 2U0/kT => a = VL sqrt(kT/2U0)
# but I have chosen before U0 = kb* 1mK so a = sqrt(T/(2)) with T in mK

sol_arr=[]

pbar = progressbar.ProgressBar(maxval=len(vzr)).start()

for i,vi in enumerate(vzr):
    vzi = vi
    vyi = vyr[i]
    vxi = vxr[i]
    pbar.update(i)
    
    zi = np.random.normal(0.0,w0/10.0)/L
    yi = np.random.normal(0.0,w0/10.0)/L
    xi = np.random.normal(0.0,w0/10.0)/L

    #print('(zi,yi,xi,vzi,vyi,vxi):',zi,yi,xi,vzi,vyi,vxi)
    
    sol = solve_ivp(af2, (tstart,tend_sim), (vzi,zi,vyi,yi,vxi,xi), t_eval = np.linspace(tstart,tend_sim,int((tend_sim-tstart)/dt),endpoint=True),method=BESTMETHOD,atol=1e-6,rtol=1e-10)#,atol=1e-6,rtol=1e-10)#, t_eval=None, dense_output=False, events=None, vectorized=False, **options)[source]
    sol_arr.append(sol)
    
end = time.time()
pbar.finish()
print('finished in (seconds):', end - start)

print('--- rk45 (DP5) ----')
print('message:', sol.message)
print('sol status:', sol.status)
if sol.status ==0:
    print('SUCCESS, reached end.')
print('sol arry', len(sol_arr))
print('sol shape:',sol.y.shape)

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_perpendicular_gaussian_transport_0_5.npz',sol_arr=sol_arr)
print('--done--')