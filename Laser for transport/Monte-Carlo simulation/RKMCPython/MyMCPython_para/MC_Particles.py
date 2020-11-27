import numpy as np

class Cesium :
    
    #Cesium Physical Properties
    mass = 2.20694657e-25 #kg #cesium mass

    #Cesium D2 Transition Optical Properties
    wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
    lambda_a = 852.34727582e-9 #m #wavelength
    gamma = 2*np.pi*5.234e6 #Hz #decay rate
    Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)