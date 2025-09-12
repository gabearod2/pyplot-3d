import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from .utils import ypr_to_R



class Sphere:
    '''
    Draws a sphere at a given position.
    '''

    def __init__(self, ax, r, c='b', x0=np.array([0, 0, 0]).T, resolution=20):
        '''
        Initialize the sphere.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            r: (float) radius of the sphere
            c: (string) color of the sphere, default 'b'
            x0: (3x1 numpy.ndarray) initial position of the sphere, default
                is [0, 0, 0]
            resolution: (int) resolution of the plot, default 20

        Returns:
            None
        '''

        self.ax = ax
        self.r = r
        self.color = c
        self.x0 = x0
        self.reso = resolution
    

    def draw(self):
        '''
        Draw the sphere with the initially defined position when the class was
        instantiated.

        Args:
            None
        
        Returns:
            None
        '''

        vertices = np.linspace(0, 2*np.pi, self.reso+1)
        u, v = np.meshgrid(vertices, vertices)

        x = self.r * np.cos(u) * np.sin(v) + self.x0[0]
        y = self.r * np.sin(u) * np.sin(v) + self.x0[1]
        z = self.r * np.cos(v) + self.x0[2]

        self.ax.plot_surface(x, y, z, color=self.color)
    

    def draw_at(self, position=np.array([0.0, 0.0, 0.0]).T):
        '''
        Draw the sphere at a given position.

        Args:
            position: (3x1 numpy.ndarray) position of the sphere, 
                default = [0.0, 0.0, 0.0]
        
        Returns:
            None
        '''

        vertices = np.linspace(0, 2*np.pi, self.reso+1)
        u, v = np.meshgrid(vertices, vertices)

        x = self.r * np.cos(u) * np.sin(v) + position[0]
        y = self.r * np.sin(u) * np.sin(v) + position[1]
        z = self.r * np.cos(v) + position[2]

        self.ax.plot_surface(x, y, z, color=self.color)



class Arrow:
    '''
    Draws an arrow at a given position, with a given attitude.
    '''

    def __init__(self, ax, direction, c='b', x0=np.array([0.0, 0.0, 0.0]).T, \
        length=1.0):
        '''
        Initialize the arrow.

        Params:
            ax: (matplotlib axis) the axis where the arrow should be drawn
            direction: (3x1 numpy.ndarray) direction of the arrow
            c: (string) color of the arrow, default = 'b'
            x0: (3x1 numpy.ndarray) origin of the arrow, 
                default = [0.0, 0.0, 0.0]
            length: (float) length of the arrow, default = 1.0

        Returns:
            None
        '''

        self.ax = ax
        self.u0 = direction
        self.color = c
        self.x0 = x0
        self.arrow_length = length
    

    def draw(self):
        '''
        Draw the arrow with the initially defined parameter when the class was
        instantiated.

        Args:
            None
        
        Returns:
            None
        '''

        x = self.x0
        u = self.u0

        self.ax.quiver(x[0], x[1], x[1], \
            u[0], u[1], u[2], \
            color=self.color,
            length=self.arrow_length, \
            normalize=False)
    

    def draw_from_to(self, x=np.array([0.0, 0.0, 0.0]).T, \
        u=np.array([1.0, 0.0, 0.0]).T):
        '''
        Draw the arrow at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) origin of the arrow, 
                default = [0.0, 0.0, 0.0]
            u: (3x1 numpy.ndarray) direction of the arrow, 
                default = [1.0, 0.0, 0.0]
        
        Returns:
            None
        '''
        
        self.ax.quiver(x[0], x[1], x[2], \
            u[0], u[1], u[2], \
            color=self.color,
            length=self.arrow_length, \
            normalize=False)



class Line:
    '''
    Draws a line at a given position, with a given attitude.
    '''

    def __init__(self, ax, c='b', x0=np.array([0.0, 0.0, 0.0]).T, \
        x1=np.array([1.0, 0.0, 0.0]).T):
        '''
        Initialize the line.
        Params:
            ax: (matplotlib axis) the axis where the line should be drawn
            c: (string) color of the arrow, default = 'b'
            x0: (3x1 numpy.ndarray) start of the line, 
                default = [0.0, 0.0, 0.0]
            x1: (3x1 numpy.ndarray) end of the line, 
                default = [1.0, 0.0, 0.0]
                
        Returns:
            None
        '''

        self.ax = ax
        self.color = c
        self.x0 = x0
        self.x1 = x1
    

    def draw(self):
        '''
        Draw the line with the initially defined parameter when the class was
        instantiated.
        Args:
            None
        
        Returns:
            None
        '''
        
        self.ax.plot([self.x0[0], self.x1[0]], \
            [self.x0[1], self.x1[1]], \
            [self.x0[2], self.x1[2]], \
            color=self.color)
    

    def draw_from_to(self, x0=np.array([0.0, 0.0, 0.0]).T, \
        x1=np.array([1.0, 0.0, 0.0]).T):
        '''
        Draw the line between two points.
        Args:
            x0: (3x1 numpy.ndarray) start of the line, 
                default = [0.0, 0.0, 0.0]
            x1: (3x1 numpy.ndarray) end of the line, 
                default = [1.0, 0.0, 0.0]
        
        Returns:
            None
        '''
        
        self.ax.plot([x0[0], x1[0]], \
            [x0[1], x1[1]], \
            [x0[2], x1[2]], \
            color=self.color)


class Plate:
    '''
    Draws a plate at a given position.
    '''

    def __init__(self, ax, r, c='b', x=np.array([0, 0, 0]).T, \
        R=np.eye(3), resolution=1):
        '''
        Initialize the plate.
        Params:
            ax: (matplotlib axis) the axis where the plane should be drawn
            r = (float): radius
            c: (string) color of the plane, default 'b'
            x: (3x1 numpy.ndarray) initial position of the plane, default
                is [0, 0, 0]
            R: (3x1 numpy.ndarray) attitude of the plane, 
                default = eye(3)
            resolution: (int) resolution of the plot, default 1
        '''
        self.ax = ax
        self.r = r
        self.color = c
        self.x = x
        self.R = R
        self.reso = resolution
        self.uvw = np.array([])
        self.mesh_shape = (1, 1)
    

    def draw(self):
        '''
        Draw the plane with the initially defined position when the class was
        instantiated.
        '''

        if self.uvw.size == 0:
            reso = self.reso
            radius = self.r 

            theta = np.linspace(0.0, 2*np.pi, reso+1)
            r     = np.linspace(0.0, radius, reso+1)
            T, R  = np.meshgrid(theta, r)  # shape: (reso+1, reso+1)
            u = R * np.cos(T)
            v = R * np.sin(T)

            w_zeros = np.zeros_like(u)

            self.mesh_shape = u.shape
            self.uvw = np.array([u.ravel(), v.ravel(), w_zeros.ravel()])

        # rotate & draw as you already do
        uvw = self.R @ self.uvw
        self.ax.plot_surface(
            uvw[0, :].reshape(self.mesh_shape) + float(self.x[0]),
            uvw[1, :].reshape(self.mesh_shape) + float(self.x[1]),
            uvw[2, :].reshape(self.mesh_shape) + float(self.x[2]),
            color=self.color,
        )


    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Draw the plane at a given position and attitude.

        Args:
            x: (3x1 numpy.ndarray) position of plane,
                default = [0.0, 0.0, 0.0]
            R: (3x1 numpy.ndarray) attitude of the plane, 
                default = eye(3)
        '''
        self.R = R
        if self.uvw.size == 0:
            reso = self.reso
            radius = self.r 

            theta = np.linspace(0.0, 2*np.pi, reso+1)
            r     = np.linspace(0.0, radius, reso+1)
            T, R  = np.meshgrid(theta, r)  # shape: (reso+1, reso+1)
            u = R * np.cos(T)
            v = R * np.sin(T)

            w_zeros = np.zeros_like(u)

            self.mesh_shape = u.shape
            self.uvw = np.array([u.ravel(), v.ravel(), w_zeros.ravel()])

        # rotate & draw as you already do
        uvw = self.R @ self.uvw
        self.ax.plot_surface(
            uvw[0, :].reshape(self.mesh_shape) + float(x[0]),
            uvw[1, :].reshape(self.mesh_shape) + float(x[1]),
            uvw[2, :].reshape(self.mesh_shape) + float(x[2]),
            color=self.color,
        )