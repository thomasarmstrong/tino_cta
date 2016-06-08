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
from mpl_toolkits.mplot3d import Axes3D



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
    
    xis1 = []
    xis2 = []
    xisb = []
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
            
            
            fit.get_great_circles(data)
            result1, crossings = fit.fit_crosses()
            result2            = fit.fit_MEst(result1)
            
            
            shower = event.mc
            # corsika measures azimuth the other way around, using phi=-az
            shower_dir = set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
            # shower direction is downwards, shower origin up
            shower_org = -shower_dir
            
            print()
            print(get_phi_theta(result1).to(u.deg) )
            print(get_phi_theta(shower_org).to(u.deg) )
            
            
            xi1 = angle(result1, shower_org).to(u.deg)
            xi2 = angle(result2, shower_org).to(u.deg)
            print("\nxi1 = {}".format( xi1 ) )
            print(  "xi2 = {}".format( xi2 ) )
            xis1.append(math.log10(xi1.value))
            xis2.append(math.log10(xi2.value))
            
            xisb.append( math.log10( min(xi1.value, xi2.value) ) )
            
            print("median1: = {} degrees"    .format(10**sorted(xis1)[ len(xis1)//2 ] ) )
            print("median2: = {} degrees"    .format(10**sorted(xis2)[ len(xis2)//2 ] ) )
            print("medianb: = {} degrees\n\n".format(10**sorted(xisb)[ len(xisb)//2 ] ) )

            #X,Y,Z = [],[],[]
            #for res in crossings:
                #X.append(res[0])
                #Y.append(res[1])
                #Z.append(res[2])
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(X,Y,Z, c='r')
            ##ax.set_aspect('equal', 'datalim')
            #ax.scatter( [shower_org[0]],[shower_org[1]],[shower_org[2]], c='b')
            #ax.scatter( [result[0]],[result[1]],[result[2]], c='g')
            #ax.set_xlim3d( shower_org[0]-.1, shower_org[0]+.1)
            #ax.set_ylim3d( shower_org[1]-.1, shower_org[1]+.1)
            #ax.set_zlim3d( shower_org[2]-.1, shower_org[2]+.1)
            #plt.show()


            if stop: break
        if stop: break  
    
    
    
    figure = plt.figure()
    plt.hist(xis1, bins=np.linspace(-3,1,50)  )
    plt.xlabel(r"log($\xi_1$ / deg)")

    figure = plt.figure()
    plt.hist(xis2, bins=np.linspace(-3,1,50)  )
    plt.xlabel(r"log($\xi_2$ / deg)")

    figure = plt.figure()
    plt.hist(np.array(xis1)-np.array(xis2), bins=np.linspace(-.5,.5,50)  )
    plt.xlabel(r"$\log(\xi_1 / \deg)-\log(\xi_2 / \deg)$")

    plt.show()