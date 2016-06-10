from itertools import combinations,permutations

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
    
    
    def fit_origin_crosses(self):
        """ calculates the origin of the gamma as the weighted average direction
            of the intersections of all great circles
        """
        
        
        crossings = []
        for perm in combinations(self.circles.values(), 2):
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
            
            

    def fit_origin_minimise(self, seed=[0,0,1], test_function=None):
        """ fits the origin of the gamma with a minimisation procedure
            this function expects that get_great_circles has been run already
            a seed should be given otherwise it defaults to "straight up"
            supperted functions to minimise are an M-estimator and the 
            negative sum of the angles to all normal vectors of the 
            great circles 
            
            Parameters:
            -----------
            seed : length-3 array
                starting point of the minimisation
            test_function : member function if this class
                either _n_angle_sum or _MEst
                defaults to _n_angle_sum if none is given
                _n_angle_sum seemingly superior to _MEst
            
            Returns:
            --------
            direction : length-3 numpy array as dimensionless quantity
                best fit for the origin of the gamma from the minimisation process
        """
        
        if test_function == None: test_function = self._n_angle_sum
        
        # using the sum of the cosines of each direction with every other direction
        # don't use the product -- with many steep angles, the product will become too small and the weight (and the whole fit) useless
        weights = [ np.sum( [ length( np.cross(A.norm,B.norm) ) for A in self.circles.values() ] ) for B in self.circles.values() ]
        
        # minimising the test function
        result = minimize( test_function, seed, args=(self.circles,weights),
                            method='BFGS', options={'disp': True}
                          ).x
            
        return np.array(result)*u.dimless
        
    def _MEst(self, origin, circles, weights=None):
        """ calculates the M-Estimator:
            a modified chi2 that becomes asymptotically linear for high values
            and is therefore less sensitive to outliers
            
            the test is performed to maximise the angles between the fit direction
            and the all the normal vectors of the great circles
            
            Parameters:
            -----------
            origin : length-3 array
                direction vector of the gamma's origin used as seed
            circles : GreatCircle array
                collection of great circles created from the camera images
            weights : array
                list of weights for each image/great circle
                
            Returns:
            --------
            MEstimator : float
                
                
            Algorithm:
            ----------
            M-Est = sum[  weight * sqrt( 2 * chi**2 ) ]
            
            
            Note:
            -----
            seemingly inferior to negative sum of angles...
            
        """
        if weights == None: weights = np.ones(len(circles))
        ang = np.array([angle(origin,circ.norm) for circ in circles.values()])
        ang[ang>np.pi/2.] = np.pi-ang[ang>np.pi/2]
        return sum( weights*np.sqrt( 2.+ (ang-np.pi/2.)**2) )
    
    def _n_angle_sum(self, origin, circles, weights=None):
        """ calculates the negative sum of the angle between the fit direction 
            and all the normal vectors of the great circles
            
            Parameters:
            -----------
            origin : length-3 array
                direction vector of the gamma's origin used as seed
            circles : GreatCircle array
                collection of great circles created from the camera images
            weights : array
                list of weights for each image/great circle
                
            Returns:
            --------
            n_sum_angles : float
                negative of the sum of the angles between the test direction
                and all normal vectors of the given great circles
        """
        if weights == None: weights = np.ones(len(circles))
        ang = np.array([angle(origin,circ.norm) for circ in circles.values()])
        ang[ang>np.pi/2.] = np.pi-ang[ang>np.pi/2]
        return -sum( weights* ang )
    
    
    
    def fit_core(self, seed=[0,0,0]):
        xdir = np.array([1,0,0])
        ydir = np.array([0,1,0])
        zdir = np.array([0,0,1])
        # the core of the shower lies on the cross section of the great circle with the horizontal plane
        # the direction of this cross section is the cross-product of the normal vectors the circle with the horizontal plane
        traces = dict([ [tel_id, np.cross( circle.norm, zdir)] for tel_id, circle in self.circles.items() ])
        
        crossings = []
        t_weight = 0.
        for pair in combinations(traces.items(), 2):
            A,B = pair[0], pair[1]
            mA = A[1][1]/A[1][0]
            mB = B[1][1]/B[1][0]
            
            Ax = self.telescopes["TelX"][A[0]-1]
            Ay = self.telescopes["TelY"][A[0]-1]
            Bx = self.telescopes["TelX"][B[0]-1]
            By = self.telescopes["TelY"][B[0]-1]

            x = ( Ay+(mA*Ax) - (By+mB*Bx) ) / (mB - mA)
            crossing = np.array([Ax,Ay,0]) + x * A[1]
            weight = length(np.cross(A[1],B[1]))
            t_weight += weight
            crossings.append(crossing*weight)
        return (sum(crossings) / t_weight)*u.m 

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
        # some weight for this circle 
        # (put e.g. uncertainty on the Hillas parameters or number of PE in here)
        self.weight = 1.
        
        
