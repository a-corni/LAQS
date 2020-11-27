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
    
#Define your beam parameters

la = 1.0e-5 #wavelength in vacum 
w0 = 1e-3 # waist 1/e^2 intensity radius
k = 2.0*np.pi/la # wave-vector
zR = np.pi*w0**2/la # Rayleigh range
z0 = 0.0 #position of maximum intensity along the z-axis
g = 9.81

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

#trap freq
wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(2*U0/(mass*zR**2))/(2*np.pi)

#simulation space-time parameters
# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) 
tL = L/vL
#distance you want to achieve
D = 0.5
coeff = [g/2, -np.sqrt(0.3/2.0), -0.5]
possible_tend = np.roots(coeff)

if possible_tend[0]<0:
    tend = possible_tend[1]/tL
else :
    tend = possible_tend[0]/tL
tstart = 0.0
#nper = 20 # nb oscill periods
#tend = nper*(1.0/wx)/tL # !!! time is in unit of tL
dt = (0.05)**2*tend # 5% period in tL unit

l = 1
p = 0

def w(z):
    return w0*np.sqrt(1.0+((z-z0)/zR)**2)

def It0(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2
    
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
        self.setWindowTitle("atom simulator (J.-B. Beguin)")
    
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
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_doughnut_5.npz',allow_pickle=True)
        print(data['sol_arr'].shape)
        self.traj0 = data['sol_arr'] #stores the trajectories
        self.nb = len(self.traj0)
        print('NUMBER OF ATOMS is:', self.nb)
        
        #print number of frames
        print('number of frames:', len(self.traj0[0].t))
        
        # LOAD TRAJECTORIES
        print("loading trajectory center of mass..")
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/center_mass_doughnut_5.npz',allow_pickle=True)
        print(data['arr_0'].shape)
        self.center_mass = data['arr_0'] #stores the trajectories
        
        self.width = 800
        self.height = 600
        # generate the intensity profile images
        self.ZL = -3*10*2 #minimum point z axis
        self.ZR = 3*10*2 #maximum point z axis
        self.YB = -30 #minimum point x axis
        self.YT = 30 #maximum point x axis
    
        zarr = np.linspace(self.center_mass[0]+self.ZL, self.center_mass[0]+self.ZR, 2*64*2,endpoint=True)  #z axis #defined with normalized length #for real axis *L
        xarr = np.linspace(self.YB, self.YT, 64*2,endpoint=True) #x axis #defined with normalizad length #for real axis *L
        
        self.counter = 0
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
        d.text((0,0), str('%.6f'%(self.center_mass[0]*L)), fill=(255,255,255,255))
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
        
        zarr = np.linspace(self.center_mass[self.lasti0%self.nb]+self.ZL, self.center_mass[self.lasti0%self.nb]+self.ZR, 2*64*2,endpoint=True)  #z axis #defined with normalized length #for real axis *L
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
        d.text((0,0), str('%.6f'%(self.center_mass[i%self.nb]*L)), fill=(255,255,255,255))
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
        fname = 'D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/doughnut/frame_nb_'+str(self.counter)+'.png'
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
        print('frame:',i)
        
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
        GL.glTranslatef(self.center_mass[self.lasti0%self.nb],-5.0,-200.0) #camera set at y = -5, x = -50
        GL.glRotatef(0.0,1.0,0.0,0.0)#camera rotated from 20° around the z axis
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
            GL.glTranslatef(self.center_mass[self.lasti0%self.nb],0.0,0.0)
            self.drawpic(self.lasti0)
            GL.glPopMatrix()
            # GL.glPushMatrix()
            # GL.glRotatef(90.0,1.0,0.0,0.0)
            # GL.glTranslatef(self.center_mass[self.lasti0%self.nb],0.0,0.0)
            # self.drawpic(self.lasti0)
            # GL.glPopMatrix()
            
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
        zval = np.arange(self.center_mass[self.lasti0%self.nb]-0.5*zr,self.center_mass[self.lasti0]+0.5*zr+dz*0.001,dz)
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