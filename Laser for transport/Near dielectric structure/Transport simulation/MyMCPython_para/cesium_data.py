#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:05:02 2013

@author: jean-baptistebeguin
"""

import numpy as np

### constants
c = 299792458 # (m/s)
pi = np.arccos(-1.)  # rad
mu0 = (4.*pi*1.E-7) # (V s / A / m)
e0 = 1./(mu0*c**2)

def nSilica(ld):

    bk = np.array([0.6961663,0.4079426,0.8974794])
    lk = np.array([0.0684043,0.1162414,9.896161]) # wavelength mum

    n = 1.

    ld = ld*1.E6

    for k in range(bk.size):
        n = n + bk[k]*ld**2/(ld**2-lk[k]**2)

    n = np.sqrt(n)

    return n

            
def CsLeKien2004():
    
    nm = 1.E-9
    MHz = 1.E6
    
    ld = np.array([852.113*nm,894.347*nm,455.528*nm,459.357*nm])
    gamma = np.array([32.76*MHz,28.7*MHz,1.88*MHz,0.8*MHz])
    g = np.array([4,2,4,2]) 
    
    return ld,gamma,g
     
def alphaStark(det):
    
    gamma = np.pi*2*5.234e6
    alpha = 0

    freq0 = 852.34727582e12-4.021776399375e9-188.4885e6+201.2871e6+251.0916e6
    omlambd = np.pi*2*np.array([freq0,freq0-201.2871e6])
    omld = np.pi*2*(freq0)+det

    g = 1
    for i in range(omlambd.size):
        alpha = alpha + gamma*(g*(1-omlambd[i]**2/omld**2)/(2.*(omld**2-omlambd[i]**2)**2+omlambd[i]**2*gamma**2))
   
    CG = 2.5# glesbcg-gordan for sigma
    alpha = CG*alpha*2.*pi*e0*c**3 

    return alpha


def alphaLeKien2004(lambd):
    
    ld, gamma, g = CsLeKien2004()

    alpha = 0. 
    
    # return angular frequency from wavelength (copied from nanofiber.py)
    def OM(l):
        try:
            r = 2.*pi*c/l
        except Exception as E:
            print('OM failed')
            print(E)
    
        return r
        
    omlambd = OM(lambd)
    
    for i in range(ld.shape[0]):
        omld = OM(ld[i])
        alpha = alpha + gamma[i]*(g[i]/(omld**2*(omld**2-omlambd**2)))
   
    alpha = alpha*pi*e0*c**3 
    return alpha

def alphaComplet(lambd):
    
    ld, gamma, g = CsLeKien2004()

    alpha = 0. 
    
    # return angular frequency from wavelength (copied from nanofiber.py)
    def OM(l):
        try:
            r = 2.*pi*c/l
        except Exception as E:
            print('OM failed')
            print(E)
    
        return r
        
    omlambd = OM(lambd)
    
    for i in range(ld.shape[0]):
        omld = OM(ld[i])
        alpha = alpha + gamma[i]*(g[i]*(1-omlambd**2/omld**2)/(2.*(omld**2-omlambd**2)**2+omlambd**2*gamma[i]**2))
   
    alpha = alpha*2.*pi*e0*c**3 
    return alpha
  
def Entry_Point():
    print('jb module')

if __name__ == '__main__':
    Entry_Point()