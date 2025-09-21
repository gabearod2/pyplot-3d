import numpy as np
from .basic import Dot3D, Line, Arrow, Plate

class Uav:
    """
    Draws a quadrotor at a given position and attitude.
    All artists are created once in __init__; draw_at only updates them.
    """

    def __init__(self, ax, color='k'):
        self.ax = ax

        # Body axes (scaled for visibility; keep your previous scaling x10)
        self.b1 = np.array([0.1, 0.0, 0.0])*3
        self.b2 = np.array([0.0, 0.1, 0.0])*3
        self.b3 = np.array([0.0, 0.0, 0.1])*3

        # Motor offsets in body frame (FR, BL, BR, FL), scaled x10 to match your code
        self.fr = np.array([ 0.04, -0.04, 0.0])*4 
        self.bl = np.array([-0.04,  0.04, 0.0])*4 
        self.br = np.array([-0.04, -0.04, 0.0])*4 
        self.fl = np.array([ 0.04,  0.04, 0.0])*4 

        # Artists 
        self.body = Plate(self.ax, r=0.05, c=color)

        self.motor1 = Plate(self.ax, r=0.1, c=color)
        self.motor2 = Plate(self.ax, r=0.1, c=color)
        self.motor3 = Plate(self.ax, r=0.1, c=color)
        self.motor4 = Plate(self.ax, r=0.1, c=color)

        self.arrow_b1 = Arrow(self.ax, c='r')
        self.arrow_b2 = Arrow(self.ax, c='g')
        self.arrow_b3 = Arrow(self.ax, c='b')

        self.arm_fr = Line(self.ax, c=color)
        self.arm_bl = Line(self.ax, c=color)
        self.arm_br = Line(self.ax, c=color)
        self.arm_fl = Line(self.ax, c=color)

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]), R=np.eye(3)):
        x = np.asarray(x).reshape(3)
        R = np.asarray(R).reshape(3, 3)

        # center marker
        self.body.draw_at(x, R)

        # rotor plates
        self.motor1.draw_at(x + R @ self.fr, R)
        self.motor2.draw_at(x + R @ self.bl, R)
        self.motor3.draw_at(x + R @ self.br, R)
        self.motor4.draw_at(x + R @ self.fl, R)

        # body axes (as lines)
        self.arrow_b1.draw_from_to(x, R @ self.b1)
        self.arrow_b2.draw_from_to(x, R @ self.b2)
        self.arrow_b3.draw_from_to(x, R @ self.b3)

        # arms
        self.arm_fr.draw_from_to(x, x + R @ self.fr)
        self.arm_bl.draw_from_to(x, x + R @ self.bl)
        self.arm_br.draw_from_to(x, x + R @ self.br)
        self.arm_fl.draw_from_to(x, x + R @ self.fl)
