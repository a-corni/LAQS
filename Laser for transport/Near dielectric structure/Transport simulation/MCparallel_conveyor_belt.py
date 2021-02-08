import multiprocessing as mp
import sys
sys.path.append('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython')
sys.path.append('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MyMCPython_para')

import numpy as np 
from MyMCPython_para import MC_3D_simulation_parallel_conveyor_belt
from MyMCPython_para import MC_intensities_conveyor_belt
import time

#simulation to run
simulation_name = "Nanofiber_conveyor_belt"

#intensity to use
Intensities = MC_intensities_conveyor_belt.get_Intensities()

#nb of particles
nbpart = 4

#nb of available processors
nbproc = mp.cpu_count()
print("Number of processors : ", nbproc)

#nb of part simulated on each processor
data = [int(nbpart/nbproc) for i in range(nbproc)]

start = time.time()

#Initialize the pool
pool = mp.Pool(nbproc) 

#launch a smaller simulation on each processor
results = pool.starmap(MC_3D_simulation_parallel_conveyor_belt.simulation_npart, [(subnbpart, Intensities) for subnbpart in data])

#Close the pool
pool.close()

#Display the time needed for the computation
end = time.time()
print(end-start)

sol_array = []
for i in range(1, nbproc):
    sol_array += results[i]
    
nb_iter = sol_array[0].y.shape[1]
nb_part = len(sol_array)
print("Nb part : ", nb_part)
print("Nb iteration :", nb_iter) 

print('--saving traj data to .npz file--')
np.savez('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/solverp0_3D_'+simulation_name+'.npz',sol_arr=sol_array)
print('--done--')