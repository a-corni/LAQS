#!/usr/bin/env python

from __future__ import unicode_literals
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, genlaguerre
import imageio
import win32ui				

from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtGui import QOpenGLFramebufferObject, QPainter, QTextDocument
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal as SIGNAL

DRAWPOT = True

try:
    from OpenGL import GL
    from OpenGL import WGL
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "points",
            "PyOpenGL must be installed to run this example.")
    sys.exit(1)
    
#name of current simulation
simulation_name = "doughnut_1_100_10"


#Fundamental Physical Constants
c = 2.99792458e8 #m.s-1
mu0 = 4*np.pi*1e-7 #N/A²
hbar = 1.054571628e-34 #J.s
kb = 1.3806504e-23 # J/K # Boltzmann constant
g = 9.81 #m.s-2

#Cesium Physical Properties
mass = 2.20694657e-25 #kg #cesium mass

#Cesium D2 Transition Optical Properties
wa = 2*np.pi*351.72571850*1e12  #Hz #frequency
lambda_a = 852.34727582e-9 #m #wavelength
gamma = 2*np.pi*5.234e6 #Hz #decay rate
Is = 16.573 #W/c² #effective far detuned saturation intensity (pi-polarized line)

#Beam Properties
delta = 10e9 #Hz
wl = wa + delta #Hz
k = wl/c #m-1
la = 2.0*np.pi/k #m # wave-vector
w0 = 1e-3 #m # waist 1/e^2 intensity radius
zR = np.pi*w0**2/la #m # Rayleigh range
z0 = 0.0 #m #position of maximum intensity along the z-axis

print("la, wl, delta ", la, wl, delta)

#Doughnut beam : described using Laguerre polynomials of parameter l=1, p=0. 
l = 1 #azimuthal index >-1
p = 0 #radial index (positive)

# Potential depth in Joules, U0 = kB*T
P0 = 100e-3 #W #Laser power
I0 = 2*P0/np.pi/w0**2 #W/m**2 #Laser intensity
Imax = I0*np.exp(-1)
U0 = 3*np.pi*c**2/2/wa**3*gamma/delta*I0
Umax = 3*np.pi*c**2/2/wa**3*gamma/delta*Imax #J #Potential depth
Tp = Umax/kb #°K #Trapping potential in Temperature 
print("Tp", Tp)

# Scattering rate
gamma_scatter_0 = U0*gamma/delta/hbar
gamma_scatter_max = Umax*gamma/delta/hbar
print("Maximum scattering rate (rad.s-1): ", gamma_scatter_max)

# Length, speed, time constants
vL = 1 #np.sqrt(2*U0/mass)
tL = 1 #np.sqrt(mass/U0)*w0/2
L = 1 #w0/np.sqrt(2)
print("L, vL, tL", L, vL, tL)

#Cesium cloud parameters
nb = 1000 # nb particles
Ti = 1e-4 #K #Initial temperature of the cloud #Doppler Temperature

#Simulation's time duration and step
#Definition of the simulation's duration
D = 0.5 #distance to achieve
coeff = [-g/2, np.sqrt(kb*Ti/mass), 0.5]
possible_tend = np.roots(coeff)
if possible_tend[0]<0:
    tend2 = possible_tend[1]/tL
else :
    tend2 = possible_tend[0]/tL
    
#First part of the simulation : dispersion of the cloud
tstart1 = 0.0
tend1 = 0.01*tend2 #1% of simulation time
tstart2 = tend1
#Simulation's time step
dt = 0.0025*tend2 # 0.25% of simulation time
#Definition of the gaussian beam

#Waist at z
def w(z):
    return w0*np.sqrt(1.0+((z-z0)/zR)**2)

#Intensity of the beam    
def It0(z,y,x):
    #Unitary intensity
    #Amplitude added in the force expression
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2
    
class Window(QWidget):
    
    def __init__(self):
        
        # get a window
        super(Window, self).__init__()
        
        # get what has to be displayed
        global addSTAT
        self.glWidget = GLWidget(self,smooth=True)
        
        # prepare the layout
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        # remove the default margin around the contained widget
        mainLayout.setContentsMargins(0,0,0,0)
        
        #display
        self.setLayout(mainLayout)
        self.setWindowTitle("Atom simulator "+simulation_name)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawText(0,0, "Tda")


class GLWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None,smooth=True):
        global L
        #super(GLWidget, self).__init__(parent)
        self.parent = parent
        #fmt = QtOpenGL.QGLFormat()
        
        # QGLFormat.__init__ (self)

        # Constructs a QGLFormat object with the following default settings:
        # Double buffer: Enabled.
        # Depth buffer: Enabled.
        # RGBA: Enabled (i.e., color index disabled).
        # Alpha channel: Disabled.
        # Accumulator buffer: Disabled.
        # Stencil buffer: Enabled.
        # Stereo: Disabled.
        # Direct rendering: Enabled.
        # Overlay: Disabled.
        # Plane: 0 (i.e., normal plane).
        # Multisample buffers: Disabled
        self.bsmooth=smooth
        fmt  = QtOpenGL.QGLFormat.defaultFormat()
        fmt.setSampleBuffers(True) # require multi sampling.. works well to fight aliasing
        #fmt.setSamples(4)
        
        # double buffer is enabled by default and swap buffer should be called automatically after a paintevent.
        super(GLWidget, self).__init__(QtOpenGL.QGLFormat(fmt), parent)

        
        self.blackbkg = QtGui.QColor.fromRgbF(0.2, 0.2, 0.2, 0.0) #Black background

        
        # LOAD TRAJECTORIES
        print("loading trajectories..")
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/MC_3D_'+simulation_name+'.npz',allow_pickle=True)
        print(data['sol_arr'].shape)
        self.traj0 = data['sol_arr'] #stores the trajectories
        self.nb = len(self.traj0)
        self.nbframes = len(self.traj0[0].t)
        print('NUMBER OF ATOMS is:', self.nb)
        
        #print number of frames
        print('number of frames:', self.nbframes)
        
        # LOAD TRAJECTORIES
        print("loading trajectory center of mass..")
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/center_mass_'+simulation_name+'.npz',allow_pickle=True)
        print(data['arr_0'].shape)
        self.center_mass = data['arr_0'] #stores the trajectories
        
        self.width = 800
        self.height = 600
        # generate the intensity profile images
        self.ZL = -0.1 #minimum point z axis
        self.ZR = 0.1 #maximum point z axis
        self.YB = -0.05 #minimum point x axis
        self.YT = 0.05 #maximum point x axis
    
        zarr = np.linspace(self.center_mass[0]+self.ZL, self.center_mass[0]+self.ZR, 2*64*2,endpoint=True)  #z axis #defined with normalized length #for real axis *L
        xarr = np.linspace(self.YB, self.YT, 64*2,endpoint=True) #x axis #defined with normalizad length #for real axis *L
        
        self.counter = self.nbframes-4
        self.texture_id = []
        self.pixels = []
    
        data = np.array([[It0((zi+z0/L)*L, xi*L,0.0) for zi in zarr] for xi in xarr]) #stores the information about the intensity profile

        # map the normalized data to colors
        # image is now RGBA
        cmap = plt.cm.viridis #hot#viridis
        norm = plt.Normalize(vmin=0.0, vmax=np.max(data)) #
        image = cmap(norm(data))

        #image = cmap(data)
        # save the image
        plt.imsave('tmp.png', image)
        
        #something misterious
        i = 0
        self.texture_id.append(i)
        
        img = Image.open('tmp.png').convert("RGBA")
        d = ImageDraw.Draw(img)
        d.text((0,0), str('%.6f'%(self.center_mass[self.counter]*L)), fill=(255,255,255,255))
        itm = np.asarray(img)
        itm2 = itm.copy()
        #itm2[:,:,0] = data[:,:]
        self.pixels.append(itm2)
        print('done')
        self.lasti0 = i
        
        # The frames are being built in different threads each 20 ms 
        
        self.frameid = 0
        self.an = 135 #initial vision angle
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.animate) 
        self.timer.start(20) 

        
    def animate(self):
        #print("OpenGL Version: ",GL.glGetString(GL.GL_VERSION))
        #print("OpenGL Vendor: ",GL.glGetString(GL.GL_VENDOR))
        #print("OpenGL Renderer: ",GL.glGetString(GL.GL_RENDERER))
        
        # build a new frame number self.frameid 
        
        self.an += 0.05

        self.counter+=1

        self.frameid+=1  

        sid = 0
        
        zarr = np.linspace(self.center_mass[self.counter%self.nbframes]+self.ZL, self.center_mass[self.counter%self.nbframes]+self.ZR, 2*64*2,endpoint=True)  #z axis #defined with normalized length #for real axis *L
        xarr = np.linspace(self.YB, self.YT, 64*2,endpoint=True) #x axis #defined with normalizad length #for real axis *L
        
        data = np.array([[It0((zi+z0/L)*L, xi*L,0.0) for zi in zarr] for xi in xarr]) #stores the information about the intensity profile

        # map the normalized data to colors
        # image is now RGBA
        cmap = plt.cm.viridis #hot#viridis
        norm = plt.Normalize(vmin=0.0, vmax=np.max(data)) #
        image = cmap(norm(data))

        #image = cmap(data)
        # save the image
        plt.imsave('tmp.png', image)
        
        #something misterious
        i = self.lasti0
        self.texture_id.append(i)
        
        img = Image.open('tmp.png').convert("RGBA")
        d = ImageDraw.Draw(img)
        d.text((0,0), str('%.6f'%(self.center_mass[self.counter%self.nbframes]*L)), fill=(255,255,255,255))
        itm = np.asarray(img)
        itm2 = itm.copy()
        #itm2[:,:,0] = data[:,:]
        self.pixels.append(itm2)
        print('done')
        self.lasti0 = i+1
        
        #MY ATTEMPT TO SAVE TO HIGHER RESOLUTION THAN SCREEN in other frame buffer
        #GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0);
        tempw = 2000
        temph = 2000
        
        #print("OpenGL MAX TEXTURE SIZE: ", GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
        #print("OpenGL VIEWPORT DIMS: ",GL.glGetIntegerv(GL.GL_VIEWPORT))
        attachment = 2 # default =< try 2 for depth
        self.fbo =  QOpenGLFramebufferObject(QtCore.QSize(tempw, temph),attachment,GL.GL_TEXTURE_2D,GL.GL_RGB8)#,GL.GL_COLOR_ATTACHMENT0)#,GL.GL_RGB)
        
        
        self.fbo.bind()
        self.resizeGL(tempw,temph)
        self.paintGL()
        
        buffer=GL.glReadPixels(0, 0, tempw, temph, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,None)
        
        #print('buffer',len(buffer), tempw,temph)
        image = Image.frombytes(mode="RGB", size=(tempw, temph), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        fname = 'D:/Users/Antoine/Documents/copenhague-1/togit/MyMCPython/doughnut_'+simulation_name+'_frame_nb_'+str(self.counter)+'.png'
        image.save(fname)
        self.fbo.release()
        
        self.resizeGL(self.width,self.height)
        
        self.update()



    def drawParticles_simplepoints0(self,color_en=False):
        
        #choose the color you want to use to color the points
        colw = QtGui.QColor.fromRgbF(1.0,1.0,1.0,0.0) #White
        colb = QtGui.QColor.fromRgbF(0.0,0.0,0.0,0.2) #Black
        colr = QtGui.QColor.fromRgbF(1.0,0.0,0.0,0.0) #Red
        colbu = QtGui.QColor.fromRgbF(0.0,0.0,1.0,0.0) #Blue
         
        #color chosen black
        self.qglColor(colw) 
        GL.glBegin(GL.GL_POINTS) #initialize points
        col = colw
        self.qglColor(col)
        
        #Working on the frame number 
        i = (self.counter) % len(self.traj0[0].t)
        print('frame:', i)
        
        #We draw the position of all the particles for this frame 
        for sol in self.traj0:
            pz = sol.y[1,i]
            py = sol.y[3,i]
            #px = sol.y[5,i]
            px = 0
            GL.glVertex3f(pz, py, px) #writes the point (pz,py,px)
        
        #We stop addind points to this frame
        GL.glEnd()
        
    # base methods of QWidget
    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    # base methods of QGLWidget (see also updateGL)
    def initializeGL(self):
    
        #Define Visualization aspects :
        
        #background color
        self.qglClearColor(self.blackbkg)
        
        #self.object = self.makeObject()
        #GL.glShadeModel(GL.GL_FLAT)
        
        #shades
        if self.bsmooth:
            GL.glShadeModel(GL.GL_SMOOTH)
        else:
            GL.glShadeModel(GL.GL_FLAT)
        

        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE)
        
        #size of the points
        GL.glPointSize(5.0)
        #GL.glEnable(GL.GL_MULTISAMPLE)
        
        
    def paintGL(self):
    
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(self.center_mass[self.counter%self.nbframes],-0.05,-2) #camera set at y = -5, x = -50
        #GL.glRotatef(20.0,1.0,0.0,0.0)#camera rotated from 20° around the z axis
        GL.glRotatef(180.0,0.0,1.0,0.0) #camera rotated from 120+0.05*frameid around the y axis 

        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

        
        # drawparticles
        GL.glPushMatrix()
        GL.glTranslatef(0.0,0.0,0.0)
        self.drawParticles_simplepoints0(color_en=True)    
        GL.glPopMatrix()
        
        
        if DRAWPOT:
            GL.glPushMatrix()
            GL.glTranslatef(self.center_mass[self.counter%self.nbframes],0.0,0.0)
            self.drawpic(self.lasti0)
            GL.glPopMatrix()
            #GL.glPushMatrix()
            #GL.glRotatef(90.0,1.0,0.0,0.0)
            #GL.glTranslatef(self.center_mass[self.counter%self.nb],0.0,0.0)
            #self.drawpic(self.lasti0)
            #GL.glPopMatrix()
            
        #self.drawGrid()

        
    def drawpic(self,i):
            
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glGenTextures(1, self.texture_id[i])
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id[i])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.pixels[i].shape[1], self.pixels[i].shape[0], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, self.pixels[i])
        #print(self.pixels.shape)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_DECAL)
        
        GL.glBegin(GL.GL_QUADS)
        #GL.glColor4f(1.0,1.0,1.0,0.04)
        GL.glTexCoord2i(0, 0)
        #GL.glColor4f(1.0,1.0,1.0,0.4)
        GL.glVertex2f(self.ZL,self.YB)
        
        GL.glTexCoord2i(1, 0)
        GL.glVertex2f(self.ZR,self.YB)
        
        GL.glTexCoord2i(1, 1)
        GL.glVertex2f(self.ZR, self.YT)
        
        GL.glTexCoord2i(0, 1)
        GL.glVertex2f(self.ZL, self.YT)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)

    def drawGrid(self):
        #draw grid
        
        # line smooth and alpha blending of the lines
        if self.bsmooth:
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT,GL.GL_NICEST)
      
        GL.glBegin(GL.GL_LINES)
        
        
        xr = 200.0
        zr = 200.0
        dx = 10.0
        dz = 10.0
        xval = np.arange(-0.5*xr, +0.5*xr+dx*0.001,dx)
        zval = np.arange(self.center_mass[self.lasti0%self.nbframes]-0.5*zr,self.center_mass[self.lasti0%self.nbframes]+0.5*zr+dz*0.001,dz)
        GL.glColor4f(1.0,1.0,1.0,0.3)
        for x in xval:
            GL.glVertex3f(zval[0],0.0,x)
            GL.glVertex3f(zval[-1],0.0,x)
        for z in zval:
            GL.glVertex3f(z,0.0,xval[0])
            GL.glVertex3f(z,0.0,xval[-1])
        GL.glEnd()        
    
        
        
    def resizeGL(self, width, height):
    
        #print('WINDOWS',width,height)
        GL.glViewport(0,0,width,height)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        # projection parallele
        # Seuls les objets min <= x <= max, min <= y <= max
        # et 0 <= z < 1.0 seront vus.
        #GL.glOrtho(0.0,width,0,height,0.0,1000.0)
        #fov = 60.0
        near = 1.0#np.max([0.001,0.0001]) # near should never be equal to zero
        far = 1000.0
        depthbufferbitlost = np.log2(far/near)
        
        self.fov = 55.0#+20.0*np.cos(0.1*self.an) # angle de vue
        # en prenant la moitie de l'angle de vue et la distance du centre de vue au premier plan de clip on calcule les coordonnes laterales qui sont mappe aux coordonnes en pixels de la fenetre
        
        xn = near*np.tan(np.deg2rad(self.fov/2.0))
        aspect = (1.0*height)/np.max([1.0,width]) # au cas ou width serait zero..
        yn = xn*aspect # to keep the aspect ratio fix
        
        #GL.glFrustum(-width*0.5,width*0.5,-height*0.5,0.5*height,near,far)
        GL.glFrustum(-xn,xn,-yn,yn,near,far)
        #print(xn,yn)
        #GL.glMultMatrix()
        #GL.glOrtho(0.0,600.0,0.0,400.0,0.001,1000.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = Window()
    window.show()
    sys.exit(app.exec_())