import numpy as np
from numpy import cos
import pyhessio
from ctapipe.io import CameraGeometry

from Histogram import nDHistogram
from Geometry import *
from guessPixDirection import *
from astropy import units as u
u.dimless = u.dimensionless_unscaled

import copy


__all__ = ["FitGammaLikelihood"]

tel_phi   = 0*u.rad
tel_theta = 0*u.rad

debug = lambda x : {}
#debug = print


"""
Energy  = 0 * u.eV    # MC energy of the shower
d       = 0 * u.m     # distance of the telescope to the shower's core
delta   = 0 * u.rad   # angle between the pixel direction and the shower direction 
rho     = 0 * u.rad   # angle between the pixel direction and the direction to the shower maximum
gamma   = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and shower maximum
npe_p_a = 0 * u.m**-2 # number of photo electrons generated per PMT area
"""
edges = []
labels = []

labels.append( "Energy" )
edges.append( [.9, 1.1]*u.TeV )

labels.append("d" )
edges.append( np.linspace(0.,1000,101)*u.m )

labels.append( "delta" )
edges.append( np.linspace(140.,180.,41)*u.degree )

labels.append( "rho" )
edges.append( np.linspace(0.,6.,61)*u.degree )

labels.append( "gamma" )
edges.append( np.concatenate( (np.linspace(0.,1.,10,False),
                                np.linspace(1.,10.,18,False),
                                np.linspace(10.,170.,40,False),
                                np.linspace(170.,180.,6,True)
                                )
                            )*u.degree
            )

class FitGammaLikelihood:
    def __init__(self, edges=edges, labels=labels):
        self.seed       = None
        self.fit_result = None
        

        self.hits = nDHistogram( edges, labels )
        self.norm = nDHistogram( edges, labels )
        self.pdf = None
        
        self.pix_dirs = dict()


    
    def fill_pdf( self, event=None, value=None, coordinates=None ):
    
        if event:
            
            Energy = event.mc.energy

            shower_dir  = SetPhiThetaR(event.mc.alt, event.mc.az, 1*u.dimless)
            shower_core = np.array([ event.mc.core_x/u.m, event.mc.core_y/u.m, 0. ]) *u.m
            
            for tel_id in  event.dl0.tels_with_data:
                
                debug("\tfilling telescope {}".format(tel_id))
                
                # assuming all pixels on one telescope have the same size...
                # TODO should not be here...
                pixel_area = pyhessio.get_pixel_area(tel_id)[0]

                # the position of the telescope in the local reference frame
                tel_pos = event.tel_pos[tel_id]

                # the direction the telescope is facing
                # TODO use actual telescope directions
                tel_dir = SetPhiThetaR(0, 0, 1*u.dimless)
                
                d = Distance(shower_core, tel_pos)
            
                if tel_id not in self.pix_dirs:
                    x, y = event.meta.pixel_pos[tel_id]
                    geom = CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
                    # TODO replace with actual pixel direction when they become available
                    self.pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, event.meta.optical_foclen[tel_id])
                    
                
                # TODO use a physical estimate for the shower max in the atmosphere, not just the direction of the pixel with the most hits...
                shower_max_dir = np.zeros(3) 
                max_npe = 0
                for pix_dir, npe in zip(self.pix_dirs[tel_id], event.mc.tel[tel_id].photo_electrons):
                    if  max_npe < npe:
                        max_npe = npe
                        shower_max_dir = pix_dir

                for pix_id, npe in enumerate( event.mc.tel[tel_id].photo_electrons ):
                    
                    npe_p_a = npe / pixel_area
                    
                    # the direction the pixel is looking in
                    pixel_dir = normalise(self.pix_dirs[tel_id][pix_id] *u.m)
                    # angle between the pixel direction and the shower direction
                    delta  = Angle(pixel_dir, shower_dir)

                    """ defining angle with respect to shower maximum """
                    rho   = Angle(pixel_dir, shower_max_dir)              # angle between the pixel direction and the direction to the shower maximum
                    
                    temp_dir  = normalise(pixel_dir - shower_max_dir)     # connecting vector between the pixel direction and the shower-max direction

                    # if the current pixel is the maximum pixel, there is no angle to be defined
                    # set it to zero
                    if Length(temp_dir) == 0:
                        gamma = 0.*u.degree
                    else:
                        gamma = Angle(shower_dir - pixel_dir * shower_dir.dot(pixel_dir), # component of the shower direction perpendicular to the telescope direction
                                        temp_dir - pixel_dir *   temp_dir.dot(pixel_dir)) # component of the connecting vector between pixel direction and
                                                                                          # shower-max direction perpendicular to the telescope direction
                                                                                      
                    self.hits.fill( npe_p_a, [Energy, d, delta, rho, gamma] )
                    self.norm.fill(       1, [Energy, d, delta, rho, gamma] )

        else:
            self.hits.fill(value, coordinates)
            self.norm.fill(value, coordinates)
            
    def normalise(self):
        
        # to not divide by zero...
        self.norm.data[self.norm.data == 0] = 0.001
        self.pdf = nDHistogram(self.hits.bin_edges, self.hits.labels)
        self.pdf.data = np.divide(self.hits.data, self.norm.data)
        self.pdf.data[self.norm.data == 0] = 0.
        # revert?
        self.norm.data[self.norm.data == 0.001] = 0.


    def write_raw(self,filename):
        np.savez_compressed(filename, hits=self.hits.data,
                                      norm=self.norm.data,
                                      axes=self.hits.bin_edges,
                                      labels=self.hits.labels
                            )
    def read_raw(self, filename):
        with np.load(filename) as data:
            self.hits = nDHistogram( data['axes'], data['labels'] )
            self.hits.data = data['hits']
            self.norm = nDHistogram( data['axes'], data['labels'] )
            self.norm.data = data['norm']

    def write_pdf(self,filename):
        if self.pdf == None:
            self.normalise()
        np.savez_compressed(filename, pdf=self.pdf.data,
                                      axes=self.hits.bin_edges,
                                      labels=self.hits.labels)
    def read_pdf(self,filename):
        with np.load(filename) as data:
            self.pdf = nDHistogram( data['axes'], data['labels'] )
            self.pdf.data = data['pdf']

            
    def set_seed(self, seed):
        self.seed = seed
    
    def fit(self):
        return self.fit_result