#!/usr/bin/env python

from __future__ import unicode_literals
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, genlaguerre

from PIL import Image
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
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
    


mass = 2.2069468e-25 # cesium mass in kg
kb = 1.38064852e-23 # boltzmann constant
U0 = 1e-3*kb # potential depth in Joules, U0 = kB*T = 1mK*kB
la = 1.e-6
w0 = 5e-6
k = 2.0*np.pi/la

zR = np.pi*w0**2/la
L = w0/np.sqrt(2.0)

vL = np.sqrt(2.0*U0/mass)
tL = L/vL
z0 = 0.0

l = 0
p = 1


def w(z):
    return w0*np.sqrt(1.0+((z-z0)/zR)**2)

def It0(z,y,x):
    r_w_z_2 = (y**2+x**2)/w(z)**2
    return (2*np.math.factorial(p)/np.pi/np.math.factorial(p+l))*(w0/w(z))**2*(2*r_w_z_2)**l*np.exp(-2*r_w_z_2)*eval_genlaguerre(p,l,2*r_w_z_2)**2
    
class Window(QWidget):
    
    def __init__(self):
        super(Window, self).__init__()

        global addSTAT
        self.glWidget = GLWidget(self,smooth=True)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        
        # remove the default margin around the contained widget
        mainLayout.setContentsMargins(0,0,0,0)
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

        
        self.blackbkg = QtGui.QColor.fromRgbF(0.2, 0.2, 0.2, 0.0)

        
        # LOAD TRAJECTORIES
        print("loading trajectories..")
        data = np.load('D:/Users/Antoine/Documents/copenhague-1/togit/gaussian_forAntoine/solverp0_3D_gaussian_l_0_p_1.npz',allow_pickle=True)
        print(data['sol_arr'].shape)
        self.traj0 = data['sol_arr']
        self.nb = len(self.traj0)
        print('NUMBER OF ATOMS is:', self.nb)
        

        # generate the intensity profile images
        self.ZL = -10*2
        self.ZR = 10*2
        self.YB = -8
        self.YT = 8
        

        zarr = np.linspace(self.ZL, self.ZR, 2*64*2,endpoint=True)
        xarr = np.linspace(self.YB, self.YT, 64*2,endpoint=True)
        
        self.counter = 0
        self.texture_id = []
        self.pixels = []

        print('number of frames:', len(self.traj0[0].t))
        
        data = np.array([[It0((zi+z0/L)*L,xi*L,0.0) for zi in zarr] for xi in xarr])
        
        cmap = plt.cm.viridis #hot#viridis
        norm = plt.Normalize(vmin=0.0, vmax=1.0)

        # map the normalized data to colors
        # image is now RGBA
        image = cmap(norm(data))
        #image = cmap(data)
        # save the image
        plt.imsave('tmp.png', image)
        
        i = 0
        self.texture_id.append(i)
        
        img = Image.open('tmp.png').convert("RGBA")
        itm = np.asarray(img)
        itm2 = itm.copy()
        #itm2[:,:,0] = data[:,:]
        self.pixels.append(itm2)
        print('done')
        self.lasti0 = i
        
        
        self.frameid = 0
        self.an = 120
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)


        
    def animate(self):
        #print("OpenGL Version: ",GL.glGetString(GL.GL_VERSION))
        #print("OpenGL Vendor: ",GL.glGetString(GL.GL_VENDOR))
        #print("OpenGL Renderer: ",GL.glGetString(GL.GL_RENDERER))
        
        self.an += 0.05

        self.counter+=1

        self.frameid+=1

        self.update()



    def drawParticles_simplepoints0(self,color_en=False):
    
        colw = QtGui.QColor.fromRgbF(1.0,1.0,1.0,0.0)
        colb = QtGui.QColor.fromRgbF(0.0,0.0,0.0,0.2)
        colr = QtGui.QColor.fromRgbF(1.0,0.0,0.0,0.0)
        colbu = QtGui.QColor.fromRgbF(0.0,0.0,1.0,0.0)

        self.qglColor(colb)
        
        GL.glBegin(GL.GL_POINTS)
        col = colb
        self.qglColor(col)

        i = (self.counter) % len(self.traj0[0].t)
        print('frame:',i)
        for sol in self.traj0:
            pz = sol.y[1,i]
            py = sol.y[3,i]
            px = sol.y[5,i]
            
            GL.glVertex3f(pz, py, px)
            
        GL.glEnd()
        
        
        
    # base methods of QWidget
    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    # base methods of QGLWidget (see also updateGL)
    def initializeGL(self):
    
    
        self.qglClearColor(self.blackbkg)
        #self.object = self.makeObject()
        #GL.glShadeModel(GL.GL_FLAT)
        if self.bsmooth:
            GL.glShadeModel(GL.GL_SMOOTH)
        else:
            GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE)
        
        GL.glPointSize(2.0)
        #GL.glEnable(GL.GL_MULTISAMPLE)
        
        
    def paintGL(self):
    
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0,-5.0,-50.0)

        GL.glRotatef(20.0,1.0,0.0,0.0)
        GL.glRotatef(self.an,0.0,1.0,0.0)
        

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

