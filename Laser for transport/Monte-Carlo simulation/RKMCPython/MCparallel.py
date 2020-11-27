import sys
sys.path.append("/home/qopt/Desktop/MyMCPython")
sys.path.append("/home/qopt/Desktop/MyMCPython/MyMCPython_para")
import multiprocessing as mp
import numpy as np 
from MyMCPython_para import MC_3D_simulation_parallel
import time

#simulation to run
simulation_name = "doughnut_1_1_20"
w0 = 1e-3 #m #waist #1/e
Tp = 1e-3 #K
delta = 20e9 #Hz

kb = 1.3806504e-23 # J/K # Boltzmann constant
U0 = kb*Tp*np.exp(1) #W.

#nb of particles
nbpart = 1000

#nb of available processors
nbproc = mp.cpu_count()
print("Number of processors : ", nbproc)

#nb of part simulated on each processor
data = [int(nbpart/nbproc) for i in range(nbproc)]

start = time.time()

#Initialize the pool
pool = mp.Pool(nbproc) 

#launch a smaller simulation on each processor
results = pool.starmap(MC_3D_simulation_parallel.simulation_npart, [(subnbpart, w0, U0, delta) for subnbpart in data])

#Close the pool
pool.close()

#Display the time needed for the computation
end = time.time()
print(end-start)

sol_array = []
for i in range( nbproc):
    sol_array += results[i]
    
nb_iter = sol_array[0].y.shape[1]
nb_part = len(sol_array)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

print('--saving traj data to .npz file--')
np.savez('/home/qopt/Desktop/MyMCPython/MC_3D_'+simulation_name+'.npz',sol_arr=sol_array)
print('--done--')