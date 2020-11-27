from matplotlib import pyplot as plt
import numpy as np
from scipy.special import genlaguerre, eval_genlaguerre
from mpl_toolkits.mplot3d import Axes3D

p=0
l=0 
def derivTx(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**(l-1)*np.exp(-2*r_w_z_2)*lag*(4*x/w(z)**2)*((l-2*r_w_z_2)*lag+4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))
    
def derivTy(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**(l-1)*np.exp(-2*r_w_z_2)*lag*(4*y/w(z)**2)*((l-2*r_w_z_2)*lag+4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))
    
def derivTz(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    lag = eval_genlaguerre(p,l,2*r_w_z_2)
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*np.exp(-2*r_w_z_2)*lag*2*(2*r_w_z_2)**l*la**2*z/np.pi**2/w(z)**4*((-(1+l)+2*r_w_z_2)*lag-4*r_w_z_2*np.polyval(np.polyder(genlaguerre(p,l)),2*r_w_z_2))
    
def E(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR))
 
def It(z,y,x):
    A = E(z,y,x)
    return np.vdot(A,A)

def I_gaussian(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2

def derivativeTx(z,y,x):
    
    def f(x):
        return It(z,y,x)
    return derivative(f, x, dx=1e-11)
    
def derivativeTy(z,y,x):

    def f(y):
        return It(z,y,x)
        
    return derivative(f, y, dx=1e-11)
    
def derivativeTz(z,y,x):

    def f(z):
        return It(z,y,x)

    return derivative(f, z, dx=10**(-10))
    
fig = plt.figure()
Z = np.linspace(-10**(-4),10**(-4),1000)
plt.plot(Z, [abs(derivativeTx(0,0,x)-derivTx(0,0,x))**2 for x in Z])
#plt.plot(Z, [abs(I_gaussian(z,0,0)-It(z,0,0))**2 for z in Z])
#plt.plot(Z, [I_gaussian(z,0,0) for z in Z])
#plt.plot(Z, [It(z,0,0) for z in Z])
#plt.plot(Z, [E(z,0,0).imag for z in Z])
#plt.plot(Z, [E(z,0,0).real for z in Z])
#plt.plot(Z, [abs(E(z,0,0))**2 for z in Z])

plt.show()