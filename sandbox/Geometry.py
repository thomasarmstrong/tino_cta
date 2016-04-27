"""
    author: Tino Michael
    e-mail: tino.michael@cea.fr
"""

import numpy as np
from math import sqrt, cos, sin, acos, atan2, pi


__all__ = [
    'vec3D',
    ]


class vec3D:
    def __init__(self, x=0.,y=0.,z=0., name=''):
        self.vec = np.array([x,y,z], dtype=np.float)
        self.name=name
        
        # some aliases to un-capitalise function names
        self.x = self.X
        self.y = self.Y
        self.z = self.Z
        self.r = self.R
        self.phi   = self.Phi
        self.theta = self.Theta
        
        self.dot    = self.Dot
        self.cross  = self.Cross
        self.dist   = self.Distance
        self.angle  = self.Angle
        self.rotate = self.rotateAroundAxis
        
    # define operators
    def __str__(self):
        # some pretty printing output
        rstr = "({:.3}, {:.3}, {:.3})".format(*self.vec)
        if self.name != '': rstr = self.name+" = "+rstr
        return rstr
    def __add__(self, vec2, name=None):
        """ returns new vec3D as the sum of @self and @vec2"""
        return vec3D( *(self.vec + vec2.vec)  )
    def __sub__(self, vec2, name=None):
        """ returns new vec3D as the differece of @self and @vec2 """
        return vec3D( *(self.vec - vec2.vec), name  )
    def __mul__(self, scalar, name=None):
        """ returns a new vec3D with every element of @self multiplied by @scalar """
        return vec3D( self.vec * scalar, name )
    def __div__(self, scalar, name=None):
        """ returns a new vec3D with every element of @self divided by @scalar """
        return vec3D( self.vec / scalar, name )
    def Dot(self, vec2):
        """ scalar product between two vec3D """
        return sum(self.vec[:] * vec2.vec[:])
    def Cross(self,vec2, name=None):
        """ cross product between two vec3D """
        [a,b,c] = self.vec
        [x,y,z] = vec2.vec
        return vec3D( b*z - c*y,
                      c*x - a*z,
                      a*y - b*x, name )

    # setter functions 
    def SetName(self, name):
        self.name = name
    def Set(self, x,y,z):
        self.vec[0] = x
        self.vec[1] = y
        self.vec[2] = z
    def SetPhiThetaR(self, phi, theta, r=1.):
        self.vec[0] = r * sin(theta)*cos(phi)
        self.vec[1] = r * sin(theta)*sin(phi)
        self.vec[2] = r * cos(theta)
    
    # getter functions
    def X(self):
        return self.vec[0]
    def Y(self):
        return self.vec[1]
    def Z(self):
        return self.vec[2]
    def Phi(self):
        return atan2(self.Y(), self.X())
    def Theta(self):
        try:
            return acos(self.Z() / self.R())
        except ValueError:
            return 0
    def R(self):
        return sqrt( self.vec[0]**2 + self.vec[1]**2 + self.vec[2]**2 )
    
    def Distance(self, vec2):
        return (self - vec2).R()
    
    def Angle(self, vec2):
        """angle between this vector and another given one"""
        cos_ph = self.dot(vec2) / ( self.R() * vec2.R() )
        # shouldn't be possible
        if abs(cos_ph) >  1: 
            # maybe just a rounding error
            if abs(cos_ph) - 1 < 1**-5:
                # if cos(angle) is +1, the angle is  0
                # if cos(angle) is -1, the angle is pi
                return pi * (1-cos_ph)/2.
            # else, something went wrong
            else: raise ValueError('cos(angle) should not be larger than 1. something went wrong. cos(angle) = {}'.format(cos_ph))
        return acos( cos_ph )

    def timesMatrix(self, mat, name=None):
        """ multiplies the vec3D with a 3x3 matrix
        
        Parameters
        ----------
        mat : 3x3 numpy array
                no check performed for proper size
                if bigger:  only the first 3x3 part gets used
                if smaller: will break with out-of-range error
                
        Result
        ------
        new vec3D as the matrix product of @self and @mat
        """
        res = np.zeros(3)
        for i in range(3):
            el = 0
            for j in range(3):
                el += mat[i,j] * self.vec[j]
            res[i] = el
        return vec3D(*res, name)

    def rotateAroundAxis(self, axis, angle, name=None):
        """ rotates @self aroun @axis with @angle in radians
            creates a rotation matrix and calls the matrix 
            multiplication method
        
        Parameters
        ---------
        axis : vec3D
               axis around which the rotation is performed
        angle : float
                angle by which @self is rotated
        
        
        Result
        ------
        rotated vec3D
        """

        axis = axis.vec
        theta = np.asarray(angle)
        axis = axis/sqrt(np.dot(axis, axis))
        a = cos(theta/2.0)
        b, c, d = -axis*sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        rot_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                               [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                               [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
        return self.timesMatrix(rot_matrix, name)

