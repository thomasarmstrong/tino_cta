from sys import exit,path
from glob import glob
import argparse

import math

import numpy as np

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.containers import MCShowerData as MCShower
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle,length


from Telescope_Mask import TelDict

import matplotlib.pyplot as plt
from matplotlib import cm

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")
from datapipe.reco.FitGammaHillas import FitGammaHillas

old=False
old=True



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
    
    
    
import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id, adc_tresh=3500):
    """
    apply basic calibration
    """
    gains = pyhessio.get_calibration(tel_id)
    
    calibrated = [ (adc0-971)*gain0 if adc0 < adc_tresh else (adc1-961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains[0],gains[1]) ]
    return np.array(calibrated)







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/Data/cta/ASTRI9/gamma/")
    parser.add_argument('-r', '--runnr',   type=str, default="*")    
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave")
    parser.add_argument('--dilate', default=False, action='store_true')
    parser.add_argument('-t', '--teltype', type=str, default="all")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist = glob( "{}*run{}*gz".format(args.indir,args.runnr ))
    if len(filenamelist) == 0: 
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
        
    fit = FitGammaHillas()
    fit.setup_geometry( *load_hessio(filenamelist[0]),phi=180*u.deg,theta=20*u.deg )
    
    signal.signal(signal.SIGINT, signal_handler)
    
    NTels = []
    EnMC  = []
    xis1  = []
    xis2  = []
    xisb  = []
    
    diffs = []
    
    allowed_tels=TelDict[args.teltype],
    for filename in sorted(filenamelist):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                     allowed_tels=range(10),
                                     max_events=args.max_events)
        
        for event in source:
            
            if len(event.dl0.tels_with_data) < 2: continue
            
            print('Scanning input file... count = {}'.format(event.count))
            print('available telscopes: {}'.format(event.dl0.tels_with_data))

            data = {}
            #for tel_id, tel in event.mc.tel.items():
                #data[tel_id] = tel.photo_electrons
            #fit.get_great_circles(data, mode="none")

            for tel_id in set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data):
                d = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id)
                data[tel_id] = d
            fit.get_great_circles(data, mode=args.mode, old=old, do_dilate=args.dilate)
            
            
            shower = event.mc
            # corsika measures azimuth the other way around, using phi=-az
            shower_dir = set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
            # shower direction is downwards, shower origin up
            shower_org = -shower_dir
            
            shower_core = np.array([shower.core_x.value, shower.core_y.value])*u.m

            
            try:
                result1, crossings = fit.fit_origin_crosses()
                result2            = fit.fit_origin_minimise(result1)
            except Exception as e:
                print(e)
                continue
            finally:
                print()
                
            NTels.append(len(fit.circles))
            EnMC.append(event.mc.energy)
            
            xi1 = angle(result1, shower_org).to(u.deg)
            xi2 = angle(result2, shower_org).to(u.deg)
            print("xi1 = {}".format( xi1 ) )
            print("xi2 = {}".format( xi2 ) )
            xis1.append(xi1)
            xis2.append(xi2)
            
            xisb.append( min(xi1, xi2) )
            
            print("xi1 res (68-percentile) = {}"    .format( sorted(xis1)[ int(len(xis1)*.68) ]) ) 
            print("xi2 res (68-percentile) = {}"    .format( sorted(xis2)[ int(len(xis2)*.68) ]) ) 
            print("xib res (68-percentile) = {}\n\n".format( sorted(xisb)[ int(len(xisb)*.68) ]) ) 


            

            #seed = sum( [ np.array([ fit.telescopes["TelX"][tel_id-1], fit.telescopes["TelY"][tel_id-1] ])
                        #for tel_id in fit.circles.keys() ]
                      #) / len(fit.circles) * u.m
            #pos_fit = fit.fit_core(seed)
            #diff = length(pos_fit[:2]-shower_core)
            #diffs.append(diff)
            #print("reco = ",pos_fit, diff)
            #print("core res (68-percentile) = {}".format( sorted(diffs)[ int(len(diffs)*.68) ] ) )
            #print()




            if stop: break
        if stop: break  
    
    
    
    #figure = plt.figure()
    #plt.hist(np.log10(xis1/u.deg), bins=np.linspace(-3,1,50)  )
    #plt.xlabel(r"log($\xi_1$ / deg)")

    #figure = plt.figure()
    #plt.hist(np.log10(xis2/u.deg), bins=np.linspace(-3,1,50)  )
    #plt.xlabel(r"log($\xi_2$ / deg)")

    #figure = plt.figure()
    #plt.hist(np.array(xis1)-np.array(xis2), bins=np.linspace(-.5,.5,50)  )
    #plt.xlabel(r"$\log(\xi_1 / \deg)-\log(\xi_2 / \deg)$")

    xi_vs_tex = {}
    unit = u.deg
    for xi, ntel in zip(xis2, NTels):
        if ntel not in xi_vs_tex:
            xi_vs_tex[ntel] = [np.log10(xi/unit)]
        else:
            xi_vs_tex[ntel].append(np.log10(xi/unit))
    
    
    #xi_vs_energy = []
    #Energy_data = np.linspace(2,8,13)
    #for e in Energy_data[:-1]:
        #xi_vs_energy.append([])
    #for en,xi in zip( EnMC, xis2 ):
        #xi_vs_energy[np.digitize(np.log10(en/u.GeV), Energy_data)-1].append(np.log(xi/unit))
    #for en in xi_vs_energy:
        #if len(en) == 0: en.append(0)

    xi_vs_energy = {}
    Energy_edges = np.linspace(2,8,13)
    Energy_centres = (Energy_edges[1:]+Energy_edges[:-1])/2.
    for en,xi in zip( EnMC, xis2 ):
        ebin = np.digitize(np.log10(en/u.GeV), Energy_edges)-1
        if Energy_centres[ebin] not in xi_vs_energy:
           xi_vs_energy[Energy_centres[ebin]]  = [np.log10(xi/unit)]
        else:
           xi_vs_energy[Energy_centres[ebin]] += [np.log10(xi/unit)]


    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot( [a for a in xi_vs_tex.values()], [a for a in xi_vs_tex.keys()], points=60, widths=.5, showextrema=True, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.subplot(212)
    plt.violinplot( [a for a in xi_vs_energy.values()], [a for a in xi_vs_energy.keys()], points=60, widths=(Energy_edges[1]-Energy_edges[0])/2., showextrema=True, showmedians=True)
    plt.xlabel("Energy / GeV")
    plt.ylabel(r"log($\xi_2$ / deg)")
    
    #heatmap, x, y = np.histogram2d(NTels, xis2, bins=[5,max(NTels)-min(NTels)+1])
    ##x,y = x[:-1],y[:-1]
    #plt.pcolor(x,y,heatmap.T,cmap=cm.hot)
    ##plt.axis([x.min(), x.max(), y.min(), y.max()])
    #plt.colorbar()
    

    plt.show()