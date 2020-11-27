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
simulation_name  = "doughnut_1_100_10"

#load the data
data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/MC_3D_' + simulation_name + '.npz',allow_pickle=True)
sol_arr = data['sol_arr']
nb_part = sol_arr.shape[0]//6
nb_iter = sol_arr.shape[1]
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

#Analysis plot
center_of_mass = sol_arr[1,:]

for i in range(1,nb_part):
    center_of_mass += sol_arr[6*i+1,:]
    
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/MC_center_mass_'+simulation_name+'.npz', center_of_mass/nb_part) #unitary units
print('Center of mass over time : saved')
