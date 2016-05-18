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
                                 max_events=args.max_events-1)


    pix_dirs = dict()
    
    Energy  = 0 * u.eV    # MC energy of the shower
    d       = 0 * u.m     # distance of the telescope to the shower's core
    delta   = 0 * u.rad   # angle between the pixel direction and the shower direction 
    rho     = 0 * u.rad   # angle between the pixel direction and the direction to the interaction vertex
    gamma   = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and vertex-direction
    npe_p_a = 0 * u.m**-2 # number of photo electrons generated per PMT area
    
    fit = FitGammaLikelihood()
    #print(fit.hits)
    #exit()
    
    
    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        #if args.tel not in event.dl0.tels_with_data: continue

        fit.fill_pdf(event=event)

    fit.write_raw("test_raw.npz")
    fit.write_pdf("test_pdf.npz")

    #test = FitGammaLikelihood([], [])
    #test.read_pdf("test.npz")

    if 0:
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
