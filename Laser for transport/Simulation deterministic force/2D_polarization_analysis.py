#!/usr/bin/env python

from __future__ import unicode_literals
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, genlaguerre
import imageio

from PIL import Image
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtGui import QOpenGLFramebufferObject
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal as SIGNAL

DRAWPOT = True

try:
    from OpenGL import GL
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "points",
            "PyOpenGL must be installed to run this example.")
    sys.exit(1)

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

c = 2.998*10**8 #m.s-1
g = 9.81

la1 = 1.0e-6 #wavelength in vacum 
w0 = 15e-6 # waist 1/e^2 intensity radius
k1 = 2.0*np.pi/la1 # wave-vector
wt1 = 2*np.pi*c/la1
zR1 = np.pi*w0**2/la1 # Rayleigh range


z0 = 0. #position of maximum intensity along the z-axis

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
wz = np.sqrt(4*U0/(mass*la1**2))/(2*np.pi)

#time to run
tstart = 0.0
tend = 1e-3/tL # !!! time is in unit of tL
dt = 0.0001*tend # 5% period in tL unit

#distance to go
D = 0.5
#variation of the frequency of the counter-propagating beam
a = np.array([[-(tend)**5/30,2*tend], [-(tend)**4/4,1]])
b = np.array([D-g*(tend)**2/3,-g*tend/2])
x = np.linalg.solve(a, b)
alpha = x[0]
beta = x[1]

def wt2(t):
    return wt1-(alpha*(2*t-tend)**4/4-(alpha*(tend)**2-g/tend)*(2*t-tend)**2/2+beta)

#Definition of the gaussian beam
#Waist at z

def w1(z):
    return w0*np.sqrt(1.0+(z/zR1)**2)
def w2(z,t):
    wt2t = wt2(t)
    zR2 = w0**2*wt2t/2/c
    return w0*np.sqrt(1.0+(z/zR2)**2)
    
#Laguerre polynomials

l = 0 #azimuthal index >-1
p = 0 #radial index (positive)

def E1(t,z,y,x):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w1(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR1)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR1))

def E2(t,z,y,x):
    wt2t = wt2(t)
    zR2 = w0**2*wt2t/2/c
    r_w_z_2 = (y**2+x**2)/w2(z,t)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w2(z,t)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR2)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR2))
    
def E_pol(t,z,y,x):
    c = 2.998*10**8 #m.s-1
    wt2t = wt2(t)
    k2 = wt2t/c
    beam1 = E1(t,z,y,x)*np.exp(-1j*k1*z)
    beam2 = E2(t,z,y,x)*np.exp(-1j*k2*z)
    return beam1*np.exp(1j*wt1*t) + np.conjugate(beam2)*np.exp(1j*wt2t*t)

def It0(t,z,y,x):
    
    A = E_pol(t,z,y,x)
    return np.vdot(A,A).real
    
time_t = np.linspace(0,tend,2000)
z_axis = np.linspace(-1e-4,1e-4,1000)
x_axis = np.linspace(-w0,w0,100)

ax1 = plt.subplot(311)
plt.plot(time_t, [wt2(t) for t in time_t])
ax1.set_title('t')
ax2 = plt.subplot(312)
plt.plot(z_axis, [It0(0,z,0,0) for z in z_axis])
ax2.set_title('z')
ax3 = plt.subplot(313)
plt.plot(x_axis, [It0(0,0,0,x) for x in x_axis])
ax3.set_title('x')
plt.show()