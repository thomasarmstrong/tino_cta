from sys import exit
from glob import glob
import argparse

from math import log10,pi

import numpy as np
from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib import cm

from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument.InstrumentDescription import load_hessio



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



energy  = []
core_x  = []
core_y  = []
azimuth = []
altitud = []
N_trigg = []

tel_x   = []
tel_y   = []
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/Data_b/cta/ASTRI9")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist = glob( args.indir )
    
    if len(filenamelist) == 0:
        print("no files found")
        exit()


    signal.signal(signal.SIGINT, signal_handler)


    (h_telescopes, h_cameras, h_optics) = load_hessio(filenamelist[0])
    Ver = 'Feb2016'
    TelVer = 'TelescopeTable_Version{}'.format(Ver)
    #CamVer = 'CameraTable_Version{}_TelID'.format(Ver)
    #OptVer = 'OpticsTable_Version{}_TelID'.format(Ver)
    
    telescopes = h_telescopes[TelVer]
    #cameras    = lambda tel_id : h_cameras[CamVer+str(tel_id)]
    #optics     = lambda tel_id : h_optics [OptVer+str(tel_id)]
    
    
    for telx, tely in zip(telescopes['TelX'],telescopes['TelY']):
        tel_x.append( telx )
        tel_y.append( tely )
    
    for filename in filenamelist:
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    # for now use only identical telescopes...
                                    #allowed_tels=TelDict[args.teltype],
                                    max_events=args.max_events)
        
        
        
        
        for event in source:
            energy .append( log10(event.mc.energy / u.TeV) )
            core_x .append( event.mc.core_x.value )
            core_y .append( event.mc.core_y.value )
            azimuth.append( event.mc.az    .value if event.mc.az.value < pi else event.mc.az.value-2*pi)
            altitud.append( event.mc.alt   .value )
            N_trigg.append( len(event.trig.tels_with_trigger))
            
            if stop: break
        if stop: break
    
    
    fig, axes = plt.subplots(2,3)
    
    ax = axes[0,0]
    ax.hist(energy )
    ax.set_xlabel(r'$\log_{10}(E/\mathrm{TeV})$')
    Nentries = len(energy)
    Emean = sum(energy)/Nentries
    ERMS  = ( sum(( np.array(energy) - Emean)**2) )**.5
    ax.text(.6,.90, "Entries:", transform = ax.transAxes)
    ax.text(.6,.85, "Mean:",    transform = ax.transAxes)
    ax.text(.6,.80, "RMS:",     transform = ax.transAxes)
    ax.text(.95,.90,                   Nentries,transform = ax.transAxes,horizontalalignment='right')
    ax.text(.95,.85, "{:10.3f}".format(Emean),  transform = ax.transAxes,horizontalalignment='right')
    ax.text(.95,.80, "{:10.3f}".format(ERMS ),  transform = ax.transAxes,horizontalalignment='right')

    axes[0,1].hist(azimuth)
    axes[0,1].set_xlabel(r'azimuth / rad')

    axes[0,2].hist(altitud)
    axes[0,2].set_xlabel(r'altitude / rad')
        
    ax = axes[1,0]
    ax.hist(N_trigg)
    ax.set_xlabel(r'$N^\mathrm{Tel}_\mathrm{trig}$')
    Nentries = len(N_trigg)
    NTrMean  = sum(N_trigg)/Nentries
    NTrRMS   = ( sum(( np.array(N_trigg) - NTrMean)**2) )**.5
    ax.text(.6,.90, "Entries:", transform = ax.transAxes)
    ax.text(.6,.85, "Mean:",    transform = ax.transAxes)
    ax.text(.6,.80, "RMS:",     transform = ax.transAxes)
    ax.text(.95,.90,                   Nentries,transform = ax.transAxes,horizontalalignment='right')
    ax.text(.95,.85, "{:10.3f}".format(NTrMean),  transform = ax.transAxes,horizontalalignment='right')
    ax.text(.95,.80, "{:10.3f}".format(NTrRMS ),  transform = ax.transAxes,horizontalalignment='right')
    
    axes[1,1].hexbin(core_x, core_y )
    axes[1,1].set_xlabel(r'$x_\mathrm{core}$')
    axes[1,1].set_ylabel(r'$y_\mathrm{core}$')
    
    axes[1,2].scatter(tel_x, tel_y)
    axes[1,2].set_xlabel(r'$x_\mathrm{tel}$')
    axes[1,2].set_ylabel(r'$y_\mathrm{tel}$')
    
    plt.tight_layout()
    plt.show()