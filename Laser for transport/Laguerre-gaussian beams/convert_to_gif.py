from PIL import Image
import glob

# Create the frames
frames = []
for i in range(1,50,1):
    fname = 'D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/doughnut/frame_nb_'+str(i)+'.png'
    new_frame = Image.open(fname)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/doughnut/doughnut_1.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=80, loop=0)