import numpy as np
from guessPixDirection import *
from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from astropy import units as u
import sys

from Geometry import *
from show_ADC_and_PE_per_event import *

from numpy import ceil
from matplotlib import pyplot as plt
from ctapipe import visualization, io


import pyhessio


from Histogram import nDHistogram
from FitGammaLikelihood import FitGammaLikelihood

filenamelist = []
filenamelist += glob("/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/gamma_20deg_180deg_run10*")
filenamelist += glob("/home/ichanmich/software/cta/datafiles/*")

filename = filenamelist[0]


tel_phi   = 0*u.rad
tel_theta = 0*u.rad
field_of_view = 2*u.degree


if __name__ == '__main__':


    


    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-t','--tel', type=int, default=1)
    parser.add_argument('-m', '--max-events', type=int, default=10000)
    parser.add_argument('-c', '--channel', type=int, default=0)
    parser.add_argument('-w', '--write', action='store_true',
                        help='write images to files')
    parser.add_argument('-s', '--show-samples', action='store_true',
                        help='show time-variablity, one frame at a time')
    parser.add_argument('--calibrate', action='store_true',
                        help='apply calibration coeffs from MC')
    args = parser.parse_args()
    
    source = hessio_event_source(filename,
                                 #allowed_tels=[args.tel],
                                 allowed_tels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                                 max_events=args.max_events)


    pix_dirs = dict()
    
    Energy  = 0 * u.eV    # MC energy of the shower
    d       = 0 * u.m     # distance of the telescope to the shower's core
    delta   = 0 * u.rad   # angle between the pixel direction and the shower direction 
    rho     = 0 * u.rad   # angle between the pixel direction and the direction to the interaction vertex
    gamma   = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and vertex-direction
    npe_p_a = 0 * u.m**-2 # number of photo electrons generated per PMT area
    
    fit = FitGammaLikelihood()
    print(fit.hits)
    exit()
    
    
    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        #if args.tel not in event.dl0.tels_with_data: continue

        fit.fill_pdf(event=event)

    #fit.write_raw("test_raw.npz")
    #fit.write_pdf("test.npz")

    #test = FitGammaLikelihood([], [])
    #test.read_pdf("test.npz")

    if 0:
        Energy = event.mc.energy

        shower_dir  = SetPhiThetaR(event.mc.alt, event.mc.az)
        shower_core = np.array([ event.mc.core_x/u.m, event.mc.core_y/u.m, 0 ]) *u.m
        shower_vert = (shower_core - shower_dir*event.mc.interaction_h)
        
        print(event.dl0.tels_with_data)

        for tel_id in  event.dl0.tels_with_data:
            pixel_area = pyhessio.get_pixel_area(tel_id)
            # the position of the telescope in the local reference frame
            tel_pos = event.tel_pos[tel_id]
            # the direction the telescope is facing
            tel_dir = normalise(SetPhiTheta(0, 0) * u.m)
            
            d = Distance(shower_core, tel_pos)
        
            # the direction in which the camera sees the vertex
            vertex_dir = normalise(shower_vert-tel_pos)
            print ("vertex_dir tel {1} = {0}".format(GetPhiTheta(vertex_dir).to(u.deg), tel_id))
        
            if tel_id not in pix_dirs:
                x, y = event.meta.pixel_pos[tel_id]

                ## mock pixel for the vertex
                #event.dl0.tel[tel_id].adc_sums[0][-1] = 8000
                #x[-1] = -2*u.m    /5.796 * 4.89
                #y[-1] = -2.77*u.m /5.796 * 4.89

                # pixel to move around to check some angles
                #event.dl0.tel[tel_id].adc_sums[0][-2] = 16000
                #x[-2] = -0.75*u.m
                #y[-2] =  0.00*u.m
                
                
                geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
                #pix_dirs[tel] = guessPixDirectionFieldView(geom.pix_x, geom.pix_y, tel_phi, tel_theta, field_of_view)
                pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, event.meta.optical_foclen[tel_id])
                
        
            shower_max_dir = np.zeros(3) 
            max_npe = 0
            for pix_dir, npe in zip(pix_dirs[tel_id], event.mc.tel[tel_id].photo_electrons):
                if  max_npe < npe:
                    max_npe = npe
                    shower_max_dir = pix_dir

            #for pix_id in range( len( event.mc.tel[tel_id].photo_electrons ) ):
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
                                 #temp_dir - tel_dir *   temp_dir.dot(tel_dir)) # component of the connecting vector between pixel-direction and 
                                                                                # vertex-direction perpendicular to the telescope direction


                """ defining angle with respect to shower maximum """
                temp_dir  = normalise(pixel_dir - shower_max_dir)      # connecting vector between the pixel direction and the shower-max direction
                rho2   = Angle(pixel_dir, shower_max_dir)              # angle between the pixel direction and the direction to the shower maximum
                gamma2 = Angle(shower_dir - pixel_dir * shower_dir.dot(pixel_dir), # component of the shower direction perpendicular to the telescope direction
                                 temp_dir - pixel_dir *   temp_dir.dot(pixel_dir)) # component of the connecting vector between pixel direction and 
                                                                                   # shower-max direction perpendicular to the telescope direction



                



                #break
                #if event.dl0.tel[tel_id].adc_sums[0][pix] == 30000 \
                #or event.dl0.tel[tel_id].adc_sums[0][pix] == 16000 \
                #or event.dl0.tel[tel_id].adc_sums[0][pix] == 8000 :
                    #print( event.dl0.tel[tel_id].adc_sums[0][pix], 
                            ##geom.pix_x[pix], geom.0  pix_y[pix],
                            #GetPhiTheta(pixel_dir).to(u.deg), 
                            #GetPhiTheta(shower_max_dir).to(u.deg), 
                            #rho2.to(u.degree), gamma2.to(u.deg) ,
                    #)
    #for event in source:                
        while True:
            response = get_input()
            print()
            if response.startswith("d"):
                disps = display_event(event, max_tel=5)
                #plt.tight_layout(pad=-0.5, w_pad=-0.9, h_pad=.0)
                plt.pause(0.1)
            elif response.startswith("p"):
                print("--event-------------------")
                print(event)
                print("--event.dl0---------------")
                print(event.dl0)
                print("--event.dl0.tel-----------")
                for teldata in event.dl0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                for tel_id in sorted(event.dl0.tel):
                    for chan in event.dl0.tel[tel_id].adc_samples:
                        npix = len(event.meta.pixel_pos[tel_id][0])
                        print("CT{:4d} ch{} pixels:{} samples:{}"
                            .format(tel_id, chan, npix,
                                    event.dl0.tel[tel_id].
                                    adc_samples[chan].shape[1]))

            elif response.startswith('q'):
                sys.exit()
            else:
                sys.exit()

        if response.startswith('q'):
            sys.exit()
