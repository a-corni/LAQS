import numpy as np
np.random.seed()

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from scipy.integrate import solve_ivp
#from scipy.stats import maxwell
from scipy.special import eval_genlaguerre
from scipy.special import genlaguerre

import progressbar

#Simulation to analyse :
simulation_name  = "Nanofiber_experiment"

#load the data
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_experiment/solverp0_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']
nb_part = sol_arr.shape
nb_iter = len(sol_arr[0].t)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

#Analysis plot
simulation_0 = sol_arr[0]
center_of_mass = simulation_0.y[1,:]

for i in range(1,nb_part[0]):
    simulation_i = sol_arr[i]
    center_of_mass += simulation_i.y[1,:]
    
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_experiment/center_mass_'+simulation_name+'.npz', center_of_mass/nb_part[0]) #unitary units
print('Center of mass over time : saved')
