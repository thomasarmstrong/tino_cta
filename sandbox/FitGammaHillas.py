from itertools import permutations

import numpy as np

from astropy import units as u

from ctapipe.io import CameraGeometry

from ctapipe.reco.hillas import hillas_parameters
from ctapipe.utils.linalg import get_phi_theta, set_phi_theta

from guessPixDirection import *

tel_phi   =   0.*u.deg
tel_theta =  20.*u.deg


class FitGammaHillas:
    
    def __init__(self):
        self.tel_geom = {}
    
    
    def set_instrument_description(self, telescopes, cameras, optics):
        self.Ver = 'Feb2016'
        self.TelVer = 'TelescopeTable_Version{}'.format(self.Ver)
        self.CamVer = 'CameraTable_Version{}_TelID'.format(self.Ver)
        self.OptVer = 'OpticsTable_Version{}_TelID'.format(self.Ver)
        
        self.telescopes = telescopes[self.TelVer]
        self.cameras    = lambda tel_id : cameras[self.CamVer+str(tel_id)]
        self.optics     = lambda tel_id : optics [self.OptVer+str(tel_id)]
    
    
    
    
    
    def fit(self, tel_data):
        
        circles = {}

        for tel_id, photo_electrons in tel_data.items():

            if tel_id not in self.tel_geom:
                from astropy import units as u # WTF???
                self.tel_geom[tel_id] = CameraGeometry.guess(self.cameras(tel_id)['PixX'].to(u.m), self.cameras(tel_id)['PixY'].to(u.m), self.telescopes['FL'][tel_id-1] * u.m) 
            moments = hillas_parameters(self.tel_geom[tel_id].pix_x, self.tel_geom[tel_id].pix_y, photo_electrons)
            
            from astropy import units as u # WTF???
            circle = GreatCircle(guessPixDirectionFocLength( np.array([ moments.cen_x.value, (moments.cen_x + moments.length * np.cos( moments.phi )).value] ) * u.m,
                                                       np.array([ moments.cen_y.value, (moments.cen_y + moments.length * np.sin( moments.phi )).value] ) * u.m,
                                                       tel_phi, tel_theta, self.telescopes['FL'][tel_id-1] * u.m
                                                     )
                                )
            circle.weight = moments.size
            circles[tel_id] = circle
        
        crossings = []
        for perm in permutations(circles.values(), 2):
            u, v, n1, a, b, n2 = perm[0].u, perm[0].v, perm[0].norm, perm[1].u, perm[1].v, perm[1].norm
            
            for indx in range(3):
                t = np.arctan2( u[indx]-a[indx], b[indx]-v[indx] )
                crossing = normalise(u * np.cos(t) + v * np.sin(t)) # * np.dot(n1,n2)
                if crossing[2] < 0: crossing *= -1
                crossings.append( crossing  )        
        result = sum(crossings) / len(crossings)
        


        
        
        
        
        




        return result
    
class GreatCircle:
    def __init__(self, dirs):
        self.u      = dirs[0]
        self.v      = dirs[1]
        self.norm   = normalise( np.cross(dirs[0], dirs[1]) )
        self.weight = 1