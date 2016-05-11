"""
    author: Tino Michael
    e-mail: tino.michael@cea.fr
"""

import numpy as np
from numpy import cos, sin, arccos as acos, arctan2 as atan2
from astropy import units as u, coordinates as coor


__all__ = [
    #'vec3D',
    'rotateAroundAxis',
    'Length',
    'Distance',
    'SetPhiThetaR',
    'SetPhiTheta',
    'GetPhiTheta',
    'normalise',
    'Angle',
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
        return vec3D( *(self.vec + vec2.vec), name  )
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
        return self.vec.dot(vec2.vec)
    def Cross(self,vec2, name=None):
        """ cross product between two vec3D """
        return vec3D( *np.cross(self.vec, vec2.vec), name)

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
        return acos( np.clip(cos_ph, -1.0, 1.0) )


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
        return self.dot(rot_matrix, name)



def rotateAroundAxis(vec, axis, angle):
    """ rotates @vec aroun @axis with @angle in radians
        creates a rotation matrix and calls the matrix 
        multiplication method
    
    Parameters
    ---------
    vec  : length-3 numpy array
            3D vector to be rotated
    axis : length-3 numpy array
            axis around which the rotation is performed
    angle : float
            angle by which @vec is rotated around @axis
    
    Result
    ------
    rotated numpy array
    """

    theta = np.asarray(angle.to(u.rad)/u.rad)
    axis = axis/(axis.dot(axis)**.5)
    a = cos(theta/2.0)
    b, c, d = -axis*sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return vec.dot(rot_matrix)

def Length(vec):
    """ returns the length / norm of a numpy array
    """
    return vec.dot(vec)**.5
    
def normalise(vec):
    """ Sets the length of the vector to 1
        without changing its direction
        
    Parameter:
    ----------
    vec : numpy array
    
    Result:
    -------
    numpy array with the same direction but length of 1
    """
    return vec / Length(vec)

def Angle(v1, v2):
    """ takes two numpy arrays and returns the angle between them
        assuming carthesian coordinates
        
    Parameters:
    -----------
    vec1 : length-3 numpy array
    vec2 : length-3 numpy array
    
    Result:
    -------
    the angle between vec1 and vec2 as a dimensioned astropy quantity
    """
    v1_u = normalise(v1)
    v2_u = normalise(v2)
    return acos(np.clip(v1_u.dot(v2_u), -1.0, 1.0))

def SetPhiThetaR(phi, theta, r=1):
    """
    """
    return np.array([ sin(theta)*cos(phi),
                      sin(theta)*sin(phi),
                      cos(theta)         ])*r
SetPhiTheta = lambda x, y: SetPhiThetaR(x,y)

def GetPhiTheta(vec):
    """
    """
    try:
        return ( atan2(vec[1], vec[0]), acos( np.clip(vec[2] / Length(vec), -1, 1) ) ) * u.rad
    except ValueError:
        return (0,0)

def Distance(vec1, vec2):
    """
    """
    return Length(vec1 - vec2)
