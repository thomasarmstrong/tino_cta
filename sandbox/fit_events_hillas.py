from sys import exit
from glob import glob
import argparse

import math

import numpy as np

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.containers import MCShowerData as MCShower
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle

from Telescope_Mask import TelDict
from FitGammaHillas import FitGammaHillas

import matplotlib.pyplot as plt




import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting after current event')
    stop = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/")
    parser.add_argument('-r', '--runnr',   type=str, default="2?")
    parser.add_argument('-t', '--teltype', type=str, default="LST")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist = glob( "{}*run{}*gz".format(args.indir,args.runnr ))
    if len(filenamelist) == 0: 
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
        
    fit = FitGammaHillas()
    fit.set_instrument_description( *load_hessio(filenamelist[0]) )
    
    signal.signal(signal.SIGINT, signal_handler)
    
    xis = []
    for filename in sorted(filenamelist):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    # for now use only identical telescopes...
                                    #allowed_tels=TelDict[args.teltype],
                                    max_events=args.max_events)
        
        for event in source:
            
            if len(event.dl0.tels_with_data) < 2: continue
            
            print('Scanning input file... count = {}'.format(event.count))
            print('available telscopes: {}'.format(event.dl0.tels_with_data))

            data = {}
            for tel_id, tel in event.mc.tel.items():
                data[tel_id] = tel.photo_electrons
            
            result = fit.fit(data)
            
            
            shower = event.mc
            shower_dir = set_phi_theta(shower.az, 90.*u.deg+shower.alt)
            shower_org = -shower_dir
            
            xi = angle(result, shower_org).to(u.deg)
            print("\nxi = {}".format( xi ) )
            xis.append(math.log10(xi.value))
            
            print("median: = {} degrees\n\n".format(10**sorted(xis)[ len(xis)//2 ] ) )

            if stop: break
        if stop: break  
    
    
    
    figure = plt.figure()
    plt.hist(xis, bins=np.linspace(-3,1,100)  )
    plt.xlabel(r"log($\xi$ / deg)")
    plt.show()