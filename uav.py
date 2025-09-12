from .basic import Sphere, Line, Arrow, Plate

import numpy as np


class Uav:
    '''
    Draws a quadrotor at a given position, with a given attitude.
    '''

    def __init__(self, ax):
        '''
        Initialize the quadrotr plotting parameters.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            arm_length: (float) length of the quadrotor arm

        Returns:
            None
        '''

        self.ax = ax

        self.b1 = np.array([0.1, 0.0, 0.0]).T*10
        self.b2 = np.array([0.0, 0.1, 0.0]).T*10
        self.b3 = np.array([0.0, 0.0, 0.1]).T*10

        self.fr = np.array([ 0.04, -0.04, 0.0]).T*10
        self.bl = np.array([-0.04,  0.04, 0.0]).T*10
        self.br = np.array([-0.04, -0.04, 0.0]).T*10
        self.fl = np.array([ 0.04,  0.04, 0.0]).T*10

        # Center of the quadrotor
        self.body = Sphere(self.ax, 0.02*10, 'k')

        # Each motor shape
        self.motor1 = Plate(self.ax, 0.02*10, 'k', resolution=50)
        self.motor2 = Plate(self.ax, 0.02*10, 'k', resolution=50)
        self.motor3 = Plate(self.ax, 0.02*10, 'k', resolution=50)
        self.motor4 = Plate(self.ax, 0.02*10, 'k', resolution=50)

        # Arrows for the each body axis
        self.arrow_b1 = Arrow(ax, self.b1, 'r')
        self.arrow_b2 = Arrow(ax, self.b2, 'g')
        self.arrow_b3 = Arrow(ax, self.b3, 'b')

        # Quadrotor arms
        self.arm_fr = Line(ax, 'k')
        self.arm_bl = Line(ax, 'k')
        self.arm_br = Line(ax, 'k')
        self.arm_fl = Line(ax, 'k')
    

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Draw the quadrotor at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) position of the center of the quadrotor, 
                default = [0.0, 0.0, 0.0]
            R: (3x3 numpy.ndarray) attitude of the quadrotor in SO(3)
                default = eye(3)
        
        Returns:
            None
        '''

        # First, clear the axis of all the previous plots
        # self.ax.clear()

        # Center of the quadrotor
        self.body.draw_at(x)

        # Each motor
        self.motor1.draw_at(x + R.dot(self.fr), R)
        self.motor2.draw_at(x + R.dot(self.bl), R)
        self.motor3.draw_at(x + R.dot(self.br), R)
        self.motor4.draw_at(x + R.dot(self.fl), R)

        # Arrows for the each body axis
        self.arrow_b1.draw_from_to(x, R.dot(self.b1))
        self.arrow_b2.draw_from_to(x, R.dot(self.b2))
        self.arrow_b3.draw_from_to(x, R.dot(self.b3))

        # Quadrotor arms
        self.arm_fr.draw_from_to(x, x + R.dot(self.fr))
        self.arm_bl.draw_from_to(x, x + R.dot(self.bl))
        self.arm_br.draw_from_to(x, x + R.dot(self.br))
        self.arm_fl.draw_from_to(x, x + R.dot(self.fl))
