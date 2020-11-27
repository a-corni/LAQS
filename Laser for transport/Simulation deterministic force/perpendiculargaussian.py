#!/usr/bin/env python

from __future__ import unicode_literals
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, genlaguerre
import imageio

from PIL import Image
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtGui import QOpenGLFramebufferObject
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal as SIGNAL

DRAWPOT = True

try:
    from OpenGL import GL
except ImportError:
    app = QtWidgets.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "points",
            "PyOpenGL must be installed to run this example.")
    sys.exit(1)

#Physical constants

c = 2.998*10**8 #m.s-1

w0 = 5e-6 # waist 1/e^2 intensity radius
z0 = 0. #position of maximum intensity along the z-axis

la1 = 1.0e-6 #wavelength in vacum 
k1 = 2.0*np.pi/la1 # wave-vector
w1 = 2*np.pi*c/la1
zR1 = np.pi*w0**2/la1 # Rayleigh range

la2 = 1.5e-6 #wavelength in vacum 
k2 = 2.0*np.pi/la2 # wave-vector
w2 = 2*np.pi*c/la2
zR2 = np.pi*w0**2/la2 # Rayleigh range

# simple gaussian/analytical (optic axis z)
mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB

# characteristic length, velocity, time scales
L = w0/np.sqrt(2.0)
vL = np.sqrt(2.0*U0/mass) #?
tL = L/vL

#trap freq
wx = np.sqrt(4*U0/(mass*w0**2))/(2*np.pi)
wz = np.sqrt(2*U0/(mass*zR1**2))/(2*np.pi)


#Definition of the gaussian beam
#Waist at z

def w1(z):
    return w0*np.sqrt(1.0+(z/zR1)**2)

def w2(z):
    return w0*np.sqrt(1.0+(z/zR2)**2)
#Laguerre polynomials

l = 0 #azimuthal index >-1
p = 0 #radial index (positive)
    
def E1(z,y,x):
    r_w_z_2 = (y**2+x**2)/w1(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w1(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR1)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR1))

def E2(z,y,x):
    r_w_z_2 = (y**2+x**2)/w2(z)**2
    return np.sqrt(2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*w0/w2(z)*(np.sqrt(2*r_w_z_2))**l*np.exp(-r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)*np.exp(-1j*r_w_z_2*z/zR2)*np.exp(1j*(l+2*p+1)*np.arctan(z/zR2))
    
def E_pol(z,y,x):
    pol1 = np.array([0,1,0])
    beam1 = E1(z,y,x)*np.exp(-1j*k1*z)*pol1
    pol2 = np.array([1,0,0])
    beam2 = E2(x,z,y)*np.exp(-1j*k2*x)
    return beam1 + np.conjugate(beam2)*pol2


def It0(z,y,x):
    
    A = E_pol(z,y,x)
    return np.vdot(A,A).real
    
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


class GLWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None,smooth=True):
        global z0
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
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_perpendiculargaussian.npz',allow_pickle=True)
        print(data['sol_arr'].shape)
        self.traj0 = data['sol_arr'] #stores the trajectories
        self.nb = len(self.traj0)
        print('NUMBER OF ATOMS is:', self.nb)
        
        #print number of frames
        print('number of frames:', len(self.traj0[0].t))
        
        self.width = 800
        self.height = 600
        # generate the intensity profile images
        self.ZL = -10*2 #minimum point z axis
        self.ZR = 10*2 #maximum point z axis
        self.YB = -8 #minimum point x axis
        self.YT = 8 #maximum point x axis
    
        zarr = np.linspace(self.ZL, self.ZR, 2*64*2,endpoint=True)  #z axis #defined with normalized length #for real axis *L
        xarr = np.linspace(self.YB, self.YT, 64*2,endpoint=True) #x axis #defined with normalizad length #for real axis *L
        
        self.counter = 0
        self.texture_id = []
        self.pixels = []
    
        data = np.array([[It0((zi+z0/L)*L,xi*L,0.0) for zi in zarr] for xi in xarr]) #stores the information about the intensity profile

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
        
    
        #MY ATTEMPT TO SAVE TO HIGHER RESOLUTION THAN SCREEN in other frame buffer
        #GL.glBindFramebufferEXT(GL.GL_FRAMEBUFFER_EXT, 0);
        tempw = 2000
        temph = 2000
        
        print("OpenGL MAX TEXTURE SIZE: ", GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
        print("OpenGL VIEWPORT DIMS: ",GL.glGetIntegerv(GL.GL_VIEWPORT))
        attachment = 2 # default =< try 2 for depth
        self.fbo =  QOpenGLFramebufferObject(QtCore.QSize(tempw, temph),attachment,GL.GL_TEXTURE_2D,GL.GL_RGB8)#,GL.GL_COLOR_ATTACHMENT0)#,GL.GL_RGB)
        
        
        self.fbo.bind()
        self.resizeGL(tempw,temph)
        self.paintGL()
        
        buffer=GL.glReadPixels(0, 0, tempw, temph, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,None)
        
        print('buffer',len(buffer), tempw,temph)
        image = Image.frombytes(mode="RGB", size=(tempw, temph), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        fname = 'D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/perpendiculargaussian/frame_nb_'+str(self.counter)+'.png'
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
        self.qglColor(colb) ###is it useful ?###
        GL.glBegin(GL.GL_POINTS) #initialize points
        col = colb
        self.qglColor(col)
        
        #Working on the frame number 
        i = (self.counter) % len(self.traj0[0].t)
        print('frame:',i)
        
        #We draw the position of all the particles for this frame 
        for sol in self.traj0:
            pz = sol.y[1,i]
            py = sol.y[3,i]
            px = sol.y[5,i]
            
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
        GL.glTranslatef(0.0,-5.0,-50.0) #camera set at y = -5, x = -50

        GL.glRotatef(20.0,1.0,0.0,0.0)#camera rotated from 20° around the z axis
        GL.glRotatef(self.an,0.0,1.0,0.0) #camera rotated from 120+0.05*frameid around the y axis 
        

        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

        
        # drawparticles
        GL.glPushMatrix()
        GL.glTranslatef(0.0,0.0,0.0)
        self.drawParticles_simplepoints0(color_en=True)    
        GL.glPopMatrix()
        
        
        i = 0
        
        if DRAWPOT:
            GL.glPushMatrix()
            GL.glTranslatef(0.0,0.0,0.0)
            self.drawpic(i)
            GL.glPopMatrix()
            GL.glPushMatrix()
            GL.glRotatef(90.0,1.0,0.0,0.0)
            GL.glTranslatef(0.0,0.0,0.0)
            self.drawpic(i)
            GL.glPopMatrix()
            
        self.drawGrid()

        
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
        xval = np.arange(-0.5*xr,0.5*xr+dx*0.001,dx)
        zval = np.arange(-0.5*zr,0.5*zr+dz*0.001,dz)
        GL.glColor4f(1.0,1.0,1.0,0.3)
        for x in xval:
            GL.glVertex3f(x,0.0,zval[0])
            GL.glVertex3f(x,0.0,zval[-1])
        for z in zval:
            GL.glVertex3f(xval[0],0.0,z)
            GL.glVertex3f(xval[-1],0.0,z)
        GL.glEnd()        
    
        
        
    def resizeGL(self, width, height):
    
        print('WINDOWS',width,height)
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

