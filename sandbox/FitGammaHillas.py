from itertools import permutations

import numpy as np

from scipy.optimize import minimize

from astropy import units as u
u.dimless = u.dimensionless_unscaled

from ctapipe.io import CameraGeometry

from ctapipe.reco.hillas import hillas_parameters
from ctapipe.utils.linalg import *

from guessPixDirection import guessPixDirection


__all__ = ["FitGammaHillas"]


class FitGammaHillas:
    
    def __init__(self):
        self.tel_geom = {}
        self.circles = {}
    
    
    def set_instrument_description(self, telescopes, cameras, optics):
        self.Ver = 'Feb2016'
        self.TelVer = 'TelescopeTable_Version{}'.format(self.Ver)
        self.CamVer = 'CameraTable_Version{}_TelID'.format(self.Ver)
        self.OptVer = 'OpticsTable_Version{}_TelID'.format(self.Ver)
        
        self.telescopes = telescopes[self.TelVer]
        self.cameras    = lambda tel_id : cameras[self.CamVer+str(tel_id)]
        self.optics     = lambda tel_id : optics [self.OptVer+str(tel_id)]
    
        self.tel_phi   =   0.*u.deg
        self.tel_theta =  20.*u.deg


    def get_great_circles(self,tel_data):
        for tel_id, photo_electrons in tel_data.items():

            if tel_id not in self.tel_geom:
                self.tel_geom[tel_id] = CameraGeometry.guess(self.cameras(tel_id)['PixX'].to(u.m),
                                                             self.cameras(tel_id)['PixY'].to(u.m),
                                                             self.telescopes['FL'][tel_id-1] * u.m) 
            moments = hillas_parameters(self.tel_geom[tel_id].pix_x,
                                        self.tel_geom[tel_id].pix_y,
                                        photo_electrons)
            
            camera_rotation = 100.893 * u.deg
            circle = GreatCircle(guessPixDirection( np.array([ moments.cen_x.value, (moments.cen_x + moments.length * np.cos( moments.psi + np.pi/2*u.rad )).value] ) * u.m,
                                                    np.array([ moments.cen_y.value, (moments.cen_y + moments.length * np.sin( moments.psi + np.pi/2*u.rad )).value] ) * u.m,
                                                    self.tel_phi, self.tel_theta, self.telescopes['FL'][tel_id-1] * u.m, camera_rotation=camera_rotation
                                                  )
                                )
            circle.weight = moments.size
            self.circles[tel_id] = circle
    
    
    def fit_crosses(self):
        crossings = []
        # this considers every combination twice...
        for perm in permutations(self.circles.values(), 2):
            n1,n2 = perm[0].norm, perm[1].norm
            # cross product automatically weighs in the angle between the two vectors
            # narrower angles have less impact, perpendicular angles have the most
            crossing = np.cross(n1,n2)
            # two great circles cross each other twice
            # (one would be the origin, the other one the direction of the gamma)
            # it doesn't matter which we pick but it should at least be consistent
            # make sure to always take the "upper" solution
            if crossing[2] < 0: crossing *= -1
            crossings.append( crossing  )
        # averaging over the solutions of all permutations
        return normalise(sum(crossings))*u.dimless, crossings
            
            
            
            
    def fit_MEst(self,seed=[0,0,1]):

        weights = [ sum( [ length( np.cross(A.norm,B.norm) ) for A in self.circles.values() ] ) for B in self.circles.values() ]

        result = minimize( self.MEst, seed, args=(self.circles,weights),
                            method='BFGS', options={'disp': True}
                          ).x
            
        return np.array(result)*u.dimless
        
    def MEst(self, origin, circles,weights):
        ang = np.array([angle(origin,circ.norm) for circ in circles.values()])
        ang[ang>np.pi/2.] = np.pi-ang[ang>np.pi/2]
        return -sum( weights*np.sqrt( 2.+ ang**2) )
    
class GreatCircle:
    """ a tiny helper class to collect some parameters for each great great circle """
    
    def __init__(self, dirs):
        """ the constructor takes two directions on the circle and creates
            the normal vector belonging to that plane
            
            Parameters:
            -----------
            dirs : shape (2,3) narray
                contains two 3D direction-vectors
                
            Algorithm:
            ----------
            c : length 3 narray
                c = (a x b) x a -> a and c form an orthogonal base for the great circle
                (only orthonormal if a and b are of unit-length)
            norm : length 3 narray
                normal vector of the circle's plane, perpendicular to a, b and c
        """
        
        self.a      = dirs[0]
        self.b      = dirs[1]
        
        # a and c form an orthogonal basis for the great circle
        # not really necessary since the norm can be calculated with a and b just as well
        self.c      = np.cross( np.cross(self.a,self.b), self.a ) 
        # normal vector for the plane the great circle is in
        self.norm   = normalise( np.cross(self.a,self.c) )
        # some weight for this circle (put e.g. uncertainty or number of PE in here)
        self.weight = 1.
        
        
