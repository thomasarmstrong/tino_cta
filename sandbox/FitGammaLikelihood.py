from Histogram import nDHistogram
from Geometry import *
from astropy import units as u
u.dimless = u.dimensionless_unscaled

class FitGammaLikelihood:
    def __init__(self):
        self.seed       = None
        self.fit_result = None
        

        """
        Energy  = 0 * u.eV   # MC energy of the shower
        npe_p_a = 0 / u.m**2 # number of photo electrons generated per PMT area
        d       = 0 * u.m    # distance of the telescope to the shower's core
        delta   = 0 * u.rad  # angle between the pixel direction and the shower direction 
        rho     = 0 * u.rad  # angle between the pixel direction and the direction to the interaction vertex
        gamma   = 0 * u.rad  # angle between shower direction and the connecting vector between pixel-direction and vertex-direction
        """
        edges = []
        labels = []
        edges.append( [.9,1., 1.1]*u.TeV )
        labels.append( "Energy" )
        edges.append( np.arange(0,1000,10)*u.m )
        labels.append("d" )
        edges.append( np.arange(0,.5,.01) )
        labels.append( "cos(delta)" )
        edges.append( np.arange(.5,1,.01) )
        labels.append( "cos(rho)" )
        edges.append( np.arange(-1,1,.1) )
        labels.append( "cos(gamma)" )
        self.pdf = nDHistogram( edges, labels )

    
    def fill_pdf( self, event=None, value=None, coordinates=None ):
    
        if event:
            
            Energy = event.mc.energy

            shower_dir  = SetPhiThetaR(event.mc.alt, event.mc.az, 1*u.dimless)
            shower_core = np.array([ event.mc.core_x/u.m, event.mc.core_y/u.m, 0 ]) *u.m
            shower_vert = (shower_core - shower_dir*event.mc.interaction_h)
            
            for tel_id in  event.dl0.tels_with_data:
                pixel_area = pyhessio.get_pixel_area(tel_id)
                # the position of the telescope in the local reference frame
                tel_pos = event.tel_pos[tel_id]
                # the direction the telescope is facing
                tel_dir = normalise(SetPhiThetaR(0, 0, 1*u.dimless))
                
                d = Distance(shower_core, tel_pos)
            
                # the direction in which the camera sees the vertex
                vertex_dir = normalise(shower_vert-tel_pos)
            
                if tel_id not in pix_dirs:
                    x, y = event.meta.pixel_pos[tel_id]
                    geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
                    pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, event.meta.optical_foclen[tel_id])
                    
            
                shower_max_dir = np.zeros(3) 
                max_npe = 0
                for pix_dir, npe in zip(pix_dirs[tel_id], event.mc.tel[tel_id].photo_electrons):
                    if  max_npe < npe:
                        max_npe = npe
                        shower_max_dir = pix_dir

                for pix_id, npe in enumerate( event.mc.tel[tel_id].photo_electrons ):
                    
                    npe_p_a = npe / pixel_area
                    
                    # the direction the pixel is seeing
                    pixel_dir = normalise(pix_dirs[tel_id][pix_id] *u.m)
                    
                    # angle between the pixel direction and the shower direction
                    delta  = Angle(pixel_dir, shower_dir)               
                    
                    
                    """ defining angles w.r.t. shower vertex """
                    #temp_dir  = normalise(pixel_dir - vertex_dir)      # connecting vector between the pixel direction and the vertex direction
                    #rho1   = Angle(pixel_dir, vertex_dir)              # angle between the pixel direction and the direction to the interaction vertex
                    #gamma1 = Angle(shower_dir - tel_dir * shower_dir.dot(tel_dir), # component of the shower direction perpendicular to the telescope direction
                                    #temp_dir - tel_dir *   temp_dir.dot(tel_dir)) # component of the connecting vector between pixel-direction and vertex-direction perpendicular to the telescope direction


                    """ defining angle with respect to shower maximum """
                    temp_dir  = normalise(pixel_dir - shower_max_dir)      # connecting vector between the pixel direction and the shower-max direction
                    rho   = Angle(pixel_dir, shower_max_dir)              # angle between the pixel direction and the direction to the shower maximum
                    gamma = Angle(shower_dir - pixel_dir * shower_dir.dot(pixel_dir), # component of the shower direction perpendicular to the telescope direction
                                    temp_dir - pixel_dir *   temp_dir.dot(pixel_dir)) # component of the connecting vector between pixel direction and shower-max direction perpendicular to the telescope direction

                    self.pdf.fill( npe_p_a, [Energy, d, delta, rho, gamma] )

        else:
            self.pdf.fill(value, coordinates)
            
            
    def set_seed(self, seed):
        self.seed = seed
    
    def fit(self):
        return self.fit_result