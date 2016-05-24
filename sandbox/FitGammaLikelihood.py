import numpy as np
import pyhessio
from ctapipe.io import CameraGeometry
from ctapipe.io.containers import MCShowerData as MCShower
from math import pi

from Histogram import nDHistogram
from Geometry import *
from guessPixDirection import *
from astropy import units as u
u.dimless = u.dimensionless_unscaled

from scipy.stats import poisson
from scipy.optimize import minimize


__all__ = ["FitGammaLikelihood"]

tel_phi   = 180.*u.rad
tel_theta =  20.*u.deg


"""
Energy  = 0 * u.eV    # MC energy of the shower
d       = 0 * u.m     # distance of the telescope to the shower's core
delta   = 0 * u.rad   # angle between the pixel direction and the shower direction 
rho     = 0 * u.rad   # angle between the pixel direction and the direction to the shower maximum
gamma   = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and shower maximum
npe_p_a = 0           # number of photo electrons generated per PMT area
probably TODO: add shower altitude (angle) as parameter
"""
edges = []
labels = []

labels.append( "Energy" )
edges.append( [.9, 1.1]*u.TeV )

labels.append("d" )
edges.append( np.linspace(0.,750,76)*u.m )

#labels.append( "delta" )
#edges.append( np.linspace(140.,180.,41)*u.degree )
labels.append( "azimuth" )
edges.append( np.linspace(50.,90.,41)*u.degree )

labels.append( "rho" )
edges.append( np.linspace(0.,6.,61)*u.degree )

labels.append( "gamma" ) # cos(gamma)?
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
        self.iteration=0

        self.hits = nDHistogram( edges, labels )
        self.norm = nDHistogram( edges, labels )
        self.pdf = None
        
        self.pix_dirs = dict()


    def get_parametrisation(self, event, shower=None, tel_data=None):
        # possibility to give another data container, not used now
        # so just prepare the photo_electrons container for more convenient loop
        if not tel_data: tel_data = dict( [ (tel_id, camera.photo_electrons) for tel_id, camera in event.mc.tel.items() ] )
        
        # if no external shower hypothesis is given, use the MC shower
        if not shower: shower = event.mc

        Energy = shower.energy

        shower_dir  = SetPhiThetaR(shower.az, 90.*u.deg+shower.alt, 1*u.dimless)
        shower_core = np.array([ shower.core_x/u.m, event.mc.core_y/u.m, 0. ]) *u.m
        

        for tel_id, photo_electrons in tel_data.items():
            
            #print("entering telescope {}".format(tel_id))
            
            # assuming all pixels on one telescope have the same size...
            # TODO should not be here...
            pixel_area = pyhessio.get_pixel_area(tel_id)[0]

            # the position of the telescope in the local reference frame
            tel_pos = event.tel_pos[tel_id]

            # the direction the telescope is facing
            # TODO use actual telescope directions
            tel_dir = SetPhiThetaR(tel_phi, tel_theta, 1*u.dimless)
            
            d = Distance(shower_core, tel_pos)
        
            if tel_id not in self.pix_dirs:
                x, y = event.meta.pixel_pos[tel_id]
                geom = CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
                # TODO replace with actual pixel direction when they become available
                self.pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, event.meta.optical_foclen[tel_id])
                
            
            # TODO use a physical estimate for the shower-max in the atmosphere, 
            # not just the direction of the pixel with the most hits...
            shower_max_dir = np.zeros(3) 
            max_npe = 0
            for pix_dir, npe in zip(self.pix_dirs[tel_id], photo_electrons):
                if  max_npe < npe:
                    max_npe = npe
                    shower_max_dir = pix_dir

            for pix_id, npe in enumerate( photo_electrons ):
                
                # the direction the pixel is looking in
                pixel_dir = normalise(self.pix_dirs[tel_id][pix_id] *u.m)

                # angle between the pixel direction and the shower direction
                #delta  = Angle(pixel_dir, shower_dir)

                # angle between the pixel direction and the direction to the shower maximum
                rho   = Angle(pixel_dir, shower_max_dir)
                
                # connecting vector between the pixel direction and the shower-max direction
                temp_dir  = normalise(pixel_dir - shower_max_dir)

                # if the current pixel contains the shower-max direction, defining an angle makes little sense
                # put the info in the underflow bin
                if Length(temp_dir)**2 < pixel_area:
                    gamma = -1.*u.degree
                else:
                    gamma = Angle(shower_dir - pixel_dir * shower_dir.dot(pixel_dir), # component of the shower direction perpendicular to the telescope direction
                                    temp_dir - pixel_dir *   temp_dir.dot(pixel_dir)) # component of the connecting vector between pixel direction and
                                                                                      # shower-max direction perpendicular to the telescope direction
                yield ([Energy, d, shower.alt, rho, gamma], npe, pixel_area)
                #yield ([Energy, d, delta, rho, gamma], npe, pixel_area)

        
    def fill_pdf( self, event=None, coordinates=None, value=None ):
    
        if event:
            # TODO add shower "max_height" to event.mc here
            for (coordinates, value, pixel_area) in self.get_parametrisation(event):
                self.hits.fill(coordinates, value/pixel_area)
                self.norm.fill(coordinates)
        else:
            self.hits.fill(coordinates, value)
            self.norm.fill(coordinates       )
            

    def normalise(self):
        
        # to not divide by zero...
        self.norm.data[self.norm.data == 0] = 0.001
        self.pdf = nDHistogram(self.hits.bin_edges, self.hits.labels)
        self.pdf.data = np.divide(self.hits.data, self.norm.data)
        #self.pdf.data[self.norm.data == 0.001] = 0. # should be zero anyway
        # revert?
        self.norm.data[self.norm.data == 0.001] = 0.



    def evaluate_pdf(self, args):
        #return self.pdf.evaluate(args)
        return self.pdf.interpolate_linear(args)


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
    
    def fit(self, event):
        
        # TODO add shower-max as soon as available in parametrisation
        shower_pars = [ 1., 3., 0., 0., 0. ]
        if self.seed:
            shower_pars[0] = self.seed.energy.to(u.TeV).value
            shower_pars[1] = self.seed.alt   .to(u.rad).value
            shower_pars[2] = self.seed.az    .to(u.rad).value
            shower_pars[3] = self.seed.core_x.to(u.m  ).value
            shower_pars[4] = self.seed.core_y.to(u.m  ).value


        # shower_pars = np.arra([ E / TeV, alt / rad, az / rad, core_x / m, core_y / m, max_height / m]) 
        fit_result = minimize( lambda x : self.get_nlog_likelihood(event, x), shower_pars, 
                              #method='nelder-mead', options={'xtol': 1e-8, 'disp': True}
                              method='BFGS', options={'disp': True}
                              )
        return fit_result
    

    def get_nlog_likelihood(self, event, shower_pars):
        self.iteration += 1
        print("get_nlog_likelihood called {}. time".format(self.iteration))
        shower = MCShower()
        shower.energy  = shower_pars[0] * u.TeV
        shower.alt     = shower_pars[1] * u.rad
        shower.az      = shower_pars[2] * u.rad
        shower.core_x  = shower_pars[3] * u.m
        shower.core_y  = shower_pars[4] * u.m
        #shower.max_height = shower_pars[5] * u.m

        log_likelihood = 0.
        for (coordinates, measured, pixel_area) in self.get_parametrisation(event, shower):
            expected = self.evaluate_pdf(coordinates)*pixel_area
            log_likelihood += max(-5.,poisson.logpmf(measured, expected))
        return -log_likelihood
#P = N**n e**-N / n!
#logP = n * logN -N - log(n!)

