import numpy as np
from matplotlib import pyplot as plt

fig, axs = plt.subplots(4)
plt.suptitle("Profile of acceleration, speed to displace atoms on a distance of D = 0.5 m")
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
print(tend)
time = [k*dt*tL for k in range(int((tend_sim-tstart)/dt))]
print(len(time))
D = 0.5
#variation of the frequency of the counter-propagating beam

def a(t):
    
    if t<=tend/4:
        return D/tend**2*(-7040/9*(t/tend)**3+320*(t/tend)**2)

    elif t<=3*tend/4  :
        return D/tend**2*(3200/9*(t/tend)**3-1600/3*(t/tend)**2+640/3*(t/tend)-160/9)

    else :
        return D/tend**2*(-7040/9*(t/tend)**3+6080/3*(t/tend)**2-5120/3*t/tend+4160/9)
        
acc = [a(t) for t in time]
axs[0].plot(time, acc)
axs[0].set(xlabel = "time (s)",  ylabel = "Acceleration (m/s²)")


def v(t):
    if t<=tend/4:
        return D/tend*(-7040/36*(t/tend)**4+320/3*(t/tend)**3)
    elif t<=3*tend/4 :
        return D/tend*(3200/36*(t/tend)**4-1600/9*(t/tend)**3+640/6*(t/tend)**2-160/9*(t/tend))-D/tend*(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4))+D/tend*(-7040/36*(1/4)**4+320/3*(1/4)**3)
    else :
        return D/tend*(-7040/36*(t/tend)**4+6080/9*(t/tend)**3-5120/6*(t/tend)**2+4160/9*t/tend)-D/tend*(-7040/36+6080/9-5120/6+4160/9)

        
speed = [v(t) for t in time]

axs[1].plot(time, speed)
axs[1].set(ylabel ="Speed (m/s)", xlabel ="time (s)")

omega = [2*k1*sp for sp in speed]

axs[2].plot(time, omega)
axs[2].set(ylabel ="Detuning dw (rad.s-1)", xlabel ="time (s)")

def pos(t):
    if t<=tend/4:
        return D*(-7040/180*(t/tend)**5+320/12*(t/tend)**4)
    elif t<=3*tend/4 :
        return D*(3200/180*(t/tend)**5-1600/36*(t/tend)**4+640/18*(t/tend)**3-160/18*(t/tend)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*t/tend)+D*(-7040/180*(1/4)**5+320/12*(1/4)**4)-D*(3200/180*(1/4)**5-1600/36*(1/4)**4+640/18*(1/4)**3-160/18*(1/4)**2+((-7040/36*(1/4)**4+320/3*(1/4)**3)-(3200/36*(1/4)**4-1600/9*(1/4)**3+640/6*(1/4)**2-160/9*(1/4)))*1/4)
    else :
        return D*(-7040/180*(t/tend)**5+6080/36*(t/tend)**4-5120/18*(t/tend)**3+4160/18*(t/tend)**2-(-7040/36+6080/9-5120/6+4160/9)*t/tend)-D*(-7040/180+6080/36-5120/18+4160/18-(-7040/36+6080/9-5120/6+4160/9))+D
 
position = [pos(t) for t in time]
 
        
axs[3].plot(time, position)
axs[3].set(ylabel ="Position (m)", xlabel = "time (s)")
plt.subplots_adjust(hspace=0.5)
plt.show()
