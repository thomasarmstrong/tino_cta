import numpy as np
import pyhessio
from ctapipe.io import CameraGeometry
from ctapipe.io.camera import _guess_camera_type
from ctapipe.io.containers import MCShowerData as MCShower
from ctapipe.utils.linalg import *
from math import pi, log, sin

from Histogram import nDHistogram
from guessPixDirection import *

from astropy.table import Table
from astropy import units as u
u.dimless = u.dimensionless_unscaled

from scipy.stats import poisson
from scipy.optimize import minimize


__all__ = ["FitGammaLikelihood"]

tel_phi   =   0.*u.rad
tel_theta =  20.*u.deg


"""
Energy   = 0 * u.eV    # MC energy of the shower
d        = 0 * u.m     # distance of the telescope to the shower's core
delta    = 0 * u.rad   # angle between the pixel direction and the shower direction 
altitude = 0 * u.deg   # angle between shower and local horizon
rho      = 0 * u.rad   # angle between the pixel direction and the direction to the shower maximum
gamma    = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and shower maximum
npe      = 0           # number of photo electrons generated on a PMT
"""
edges = []
labels = []

#labels.append( "Energy" )
#edges.append( [.9, 1.1]*u.TeV )

labels.append( "alpha" )  # angle between pixel-direction direction and telescope-direction
edges.append( np.linspace(0,4,25)*u.deg )

labels.append("d" )
edges.append( np.linspace(0,800,41)*u.m )

#labels.append( "delta" )
#edges.append( np.linspace(155,180,26)*u.degree )

#labels.append( "altitude" )
#edges.append( np.linspace(50,90,41)*u.degree )

labels.append( "rho" )
edges.append( np.linspace(0,6,31)*u.degree )

labels.append( "gamma" )
edges.append( np.concatenate( (
                               np.linspace(  0, 10, 10,False),
                               np.linspace( 10,160, 30,False),
                               np.linspace(170,180,  6,True)
                              )
                            )*u.degree
            )
from numba import jit
class FitGammaLikelihood:
    def __init__(self, edges=edges, labels=labels):
        self.seed       = None
        self.iteration=0

        self.pix_dirs = {}

        self.hits = nDHistogram( edges, labels )
        self.norm = nDHistogram( edges, labels )
        self.pdf  = None
        

    def set_instrument_description(self, telescopes, cameras, optics):
        self.Ver = 'Feb2016'
        self.TelVer = 'TelescopeTable_Version{}'.format(self.Ver)
        self.CamVer = 'CameraTable_Version{}_TelID'.format(self.Ver)
        self.OptVer = 'OpticsTable_Version{}_TelID'.format(self.Ver)
        
        self.telescopes = telescopes[self.TelVer]
        self.cameras    = lambda tel_id : cameras[self.CamVer+str(tel_id)]
        self.optics     = lambda tel_id : optics [self.OptVer+str(tel_id)]
    
    def set_atmosphere(self, filename):
        altitude  = []
        thickness = []
        atm_file = open(filename, "r")
        for line in atm_file:
            if line.startswith("#"): continue
            altitude .append(float(line.split()[0]))
            thickness.append(float(line.split()[2]))
        
        self.atmosphere = nDHistogram( [np.array(altitude)*u.km], ["altitude"] )
        self.atmosphere.data = (thickness[0:1]+thickness)*u.g * u.cm**-2

        
    def find_shower_max_height(self,energy,h_first_int,gamma_alt):
        # offset of the shower-maximum in radiation lengths
        c = 0.97 * log(energy / (83 * u.MeV)) - 1.32
        # radiation length in dry air at 1 atm = 36,62 g / cm**2 [PDG]
        c *= 36.62*u.g * u.cm**-2
        # showers with a more horizontal direction spend more path length in each atm. layer
        # the "effective transverse thickness" they have to pass is reduced
        c *= sin(gamma_alt)
        
        # find the thickness at the height of the first interaction
        t_first_int = 0.
        for ii, height1 in enumerate(self.atmosphere.bin_edges[0]):
            if h_first_int < height1:
                height2 = self.atmosphere.bin_edges[0][ii-1]
                thick1  = self.atmosphere.evaluate([height1])
                thick2  = self.atmosphere.evaluate([height2])
                
                t_first_int = (thick2-thick1) / (height2-height1) * (h_first_int.to(u.km) - height1) + thick1
                break

        # total thickness at shower maximum = thickness at first interaction + thickness traversed to shower maximum
        t_shower_max = t_first_int + c
        
        # now find the height with the wanted thickness
        for ii, thick1 in enumerate(self.atmosphere.data):
            if t_shower_max > thick1:
                height1 = self.atmosphere.bin_edges[0][ii-1]
                height2 = self.atmosphere.bin_edges[0][ii-2]
                thick2  = self.atmosphere.evaluate([height2])
                
                return (height2-height1) / (thick2-thick1) * (t_shower_max-thick1) + height1


    #@jit    
    def get_parametrisation(self, shower, tel_data):
        
        Energy = shower.energy

        shower_dir  = set_phi_theta_r(shower.az, 90.*u.deg+shower.alt, 1*u.dimless)
        shower_core = np.array([ shower.core_x/u.m, shower.core_y/u.m, 0. ]) *u.m
        shower_max_pos = shower_core - shower_dir * shower.h_shower_max / sin(shower.alt)
        

        for tel_id, photo_electrons in tel_data.items():
            
            print("entering telescope {}".format(tel_id))
            
            # if all telescopes until id are present, idx should be id-1, but to be sure
            tel_idx = np.searchsorted( self.telescopes['TelID'], tel_id )
            

            # the position of the telescope in the local reference frame
            tel_pos = np.array([  self.telescopes['TelX'][tel_idx],  self.telescopes['TelY'][tel_idx],  self.telescopes['TelZ'][tel_idx] ]) * u.m

            # the direction the telescope is facing
            # TODO use actual telescope directions
            tel_dir = set_phi_theta_r(tel_phi, tel_theta, 1*u.dimless)
            
            d = length(shower_core - tel_pos)

            # TODO replace with actual pixel direction when they become available
            if tel_id not in self.pix_dirs:
                # doesn't seem to be right for camera types with 0 degree rotation...
                # use default value of -100.893 degrees in guessPixDirectionFocLength 
                #camera_rotation = _guess_camera_type(len(self.cameras(tel_id)['PixX']), self.telescopes['FL'][tel_idx]*u.m)[4]
                geom = CameraGeometry.guess(self.cameras(tel_id)['PixX'].to(u.m), self.cameras(tel_id)['PixY'].to(u.m), self.telescopes['FL'][tel_idx] * u.m)
                self.pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, self.telescopes['FL'][tel_idx] * u.m)

            shower_max_dir = normalise(shower_max_pos-tel_pos)
            
            for pix_id, npe in enumerate( photo_electrons ):
                
                pixel_area = self.cameras(tel_id)['PixA'][pix_id]
                
                # the direction the pixel is looking in
                pixel_dir = self.pix_dirs[tel_id][pix_id]

                # angle between pixel direction and telescope diretion (for acceptance test)
                alpha = angle(pixel_dir, tel_dir)

                # angle between the pixel direction and the shower direction
                delta  = angle(pixel_dir, shower_dir)

                # angle between the pixel direction and the direction to the shower maximum
                rho   = angle(pixel_dir, shower_max_dir)
                
                # connecting vector between the pixel direction and the shower-max direction
                temp_dir  = normalise(pixel_dir - shower_max_dir)

                # if the current pixel contains the shower-max direction, defining an angle makes little sense
                # put the info in the underflow bin
                gamma = angle(shower_dir - pixel_dir * shower_dir.dot(pixel_dir), # component of the shower direction perpendicular to the telescope direction
                                temp_dir - pixel_dir *   temp_dir.dot(pixel_dir)) # component of the connecting vector between pixel direction and
                                                                                  # shower-max direction perpendicular to the telescope direction
                                                                                  
                #print ([Energy, d, delta, shower.alt, rho, gamma], npe, pixel_area)
                #yield ([Energy, d, delta, shower.alt, rho, gamma], npe, pixel_area)
                #yield ([angle(tel_dir,shower_max_dir), d, delta, shower.alt, rho, gamma], npe, pixel_area)
                yield ([ alpha, d, delta, rho, gamma], npe, pixel_area)

        
    def fill_pdf( self, event=None, coordinates=None, value=None ):
    
        if event:
            # if not done yet, add the height of the shower maximum as a field to the shower container
            try:
                event.mc.add_item("h_shower_max")
            except AttributeError:
                pass
            # find the height of the shower maximum and set it in the shower container
            event.mc.h_shower_max = self.find_shower_max_height(event.mc.energy, event.mc.h_first_int, event.mc.alt)
            
            data = dict( [ (tel_id, tel.photo_electrons) for tel_id, tel in event.mc.tel.items() ] )
            for (coordinates, value, pixel_area) in self.get_parametrisation( event.mc, data ):
                self.hits.fill(coordinates, value)
                self.norm.fill(coordinates, pixel_area)
        else:
            self.hits.fill(coordinates, value)
            self.norm.fill(coordinates       )
            

    def normalise(self):
        
        # to not divide by zero...
        self.norm.data[self.norm.data == 0] = 0.001
        self.pdf = nDHistogram(self.hits.bin_edges, self.hits.labels)
        self.pdf.data = np.divide(self.hits.data, self.norm.data)
        # revert?
        #self.norm.data[self.norm.data == 0.001] = 0.
        del self.hits.data, self.norm.data


    def evaluate_pdf(self, args):
        #return self.pdf.evaluate(args)
        return self.pdf.interpolate_linear(args)


    def write_raw(self,filename):
        np.savez_compressed(filename+"_raw.npz", axes   = self.hits.bin_edges,
                                                 hits   = self.hits.data,
                                                 norm   = self.norm.data,
                                                 labels = self.hits.labels
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
        np.savez_compressed(filename+"_pdf.npz", pdf    = self.pdf.data,
                                                 axes   = self.hits.bin_edges,
                                                 labels = self.hits.labels)
    def read_pdf(self,filename):
        with np.load(filename) as data:
            self.pdf = nDHistogram( data['axes'], data['labels'] )
            self.pdf.data = data['pdf']

            
    def fit(self, event):
        
        shower_pars = [ 1., 1.2, 0., 0., 0., 10000 ]
        if self.seed:
            shower_pars[0] = self.seed.energy      .to(u.TeV).value
            shower_pars[1] = self.seed.alt         .to(u.rad).value
            shower_pars[2] = self.seed.az          .to(u.rad).value
            shower_pars[3] = self.seed.core_x      .to(u.m  ).value
            shower_pars[4] = self.seed.core_y      .to(u.m  ).value
            shower_pars[5] = self.seed.h_shower_max.to(u.m  ).value

        
        # selct only pixel with data and those around them
        data = dict()
        for tel_id, tel in event.mc.tel.items():
            geom = CameraGeometry.guess(self.cameras(tel_id)['PixX'].to(u.m), self.cameras(tel_id)['PixY'].to(u.m), self.telescopes['FL'][tel_idx] * u.m)
            mask = tel.photo_electrons>0
            for pixid in geom.pix_id[mask]:
                mask[geom.neighbors[pixid]] = True
            data[tel_id] = tel.photo_electrons[mask]

        #data = dict( [ (tel_id, tel.photo_electrons[tel.photo_electrons>0]) for tel_id, tel in event.mc.tel.items() ] )

        # shower_pars = np.arra([ E / TeV, alt / rad, az / rad, core_x / m, core_y / m, h_shower_max / m]) 
        fit_result = minimize( lambda x : self.get_nlog_likelihood(x, data), shower_pars, 
                              #method='nelder-mead', options={'xtol': 1e-8, 'disp': True}
                              method='BFGS', options={'disp': True}
                              )
        return fit_result
    

    def get_nlog_likelihood(self, shower_pars, data):
        self.iteration += 1
        print("get_nlog_likelihood called {}. time".format(self.iteration))
        self.seed.energy       = shower_pars[0] * u.TeV
        self.seed.alt          = shower_pars[1] * u.rad
        self.seed.az           = shower_pars[2] * u.rad
        self.seed.core_x       = shower_pars[3] * u.m
        self.seed.core_y       = shower_pars[4] * u.m
        self.seed.h_shower_max = shower_pars[5] * u.m

        log_likelihood = 0.
        for (coordinates, measured, pixel_area) in self.get_parametrisation(self.seed, data):
            expected = self.evaluate_pdf(coordinates)*pixel_area
            log_likelihood += max(-200.,poisson.logpmf(measured, expected))
        return -log_likelihood
#P = N**n e**-N / n!
#logP = n * logN -N - log(n!)

