import numpy as np
from scipy import constants as cst
from PIL import Image
import glob

# Create the frames
frames_intensity = []
frames_potential = []

lam = 980e-9
k = 2*np.pi/lam
f = cst.c/lam
T = 1/f
w = 2*np.pi*f
time = [1e-10/2/np.pi*0.05*i for i in range(20)]
w1 = w

for i in range(20):
    t = time[i]
    fname_intensity = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/Total_light_intensity_time_'+str(i)+'.png'
    fname_potential = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/Nanofiber_Trap_potential_'+str(i)+'.png'
    new_frame_intensity = Image.open(fname_intensity)
    new_frame_potential = Image.open(fname_potential)
    
    frames_intensity.append(new_frame_intensity)
    frames_potential.append(new_frame_potential)
    
# for i in range(10,20):
#     fname = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_meep-out/ez-0000'+str(i)+'.00.png'
#     new_frame = Image.open(fname)
#     frames.append(new_frame)

# Save into a GIF file that loops forever
frames_intensity[0].save('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/intensity.gif', format='GIF',
               append_images=frames_intensity[1:],
               save_all=True,
               duration = 600, loop=0)
    
frames_potential[0].save('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/nanofiber_conveyor_belt/potential.gif', format='GIF',
               append_images = frames_potential[1:],
               save_all=True,
               duration = 600, loop=0)