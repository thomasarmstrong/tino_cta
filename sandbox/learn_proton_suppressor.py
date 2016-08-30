from sys import exit
from glob import glob
import argparse

from itertools import chain

from math import log10

import numpy as np

from astropy import units as u

from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils import linalg
from ctapipe.utils.fitshistogram import Histogram

from ctapipe.reco.hillas import hillas_parameters
from ctapipe.reco.cleaning import tailcuts_clean, dilate

from Telescope_Mask import TelDict

import matplotlib.pyplot as plt
from matplotlib import cm

from random import gauss

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting current loop after this event')
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
                        default="/local/home/tmichael/Data/cta/ASTRI9")
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('-t', '--teltype', type=str, default="SST_ASTRI")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist_gamma  = glob( "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    filenamelist_proton = glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))
    
    print(  "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    signal.signal(signal.SIGINT, signal_handler)


    mean_widths     = Histogram(initFromFITS="widths.fits")
    mean_widths_sq  = Histogram(initFromFITS="widths_sq.fits")
    mean_lengths    = Histogram(initFromFITS="lengths.fits")
    mean_lengths_sq = Histogram(initFromFITS="lengths_sq.fits")

    axisNames = [ "log(E / GeV)" ]
    ranges    = [ [2,8] ]
    nbins     = [ 8 ]
    wrong = { "p":Histogram( axisNames=axisNames, nbins=nbins, ranges=ranges), "g":Histogram( axisNames=axisNames, nbins=nbins, ranges=ranges) }
    total = { "p":Histogram( axisNames=axisNames, nbins=nbins, ranges=ranges), "g":Histogram( axisNames=axisNames, nbins=nbins, ranges=ranges) }
    
    
    
    (h_telescopes, h_cameras, h_optics) = load_hessio(filenamelist_gamma[0])
    Ver = 'Feb2016'
    TelVer = 'TelescopeTable_Version{}'.format(Ver)
    CamVer = 'CameraTable_Version{}_TelID'.format(Ver)
    OptVer = 'OpticsTable_Version{}_TelID'.format(Ver)
    
    telescopes = h_telescopes[TelVer]
    cameras    = lambda tel_id : h_cameras[CamVer+str(tel_id)]
    optics     = lambda tel_id : h_optics [OptVer+str(tel_id)]

    tel_phi   =   0.*u.deg
    tel_theta =  20.*u.deg
    

    Features = { "p":[], "g":[] }
    Classes  = { "p":[], "g":[] }
    MCEnergy = { "p":[], "g":[] }
    
    red_width = { "p":[], "g":[] }
    red_lenth = { "p":[], "g":[] }

    tel_geom = {}
    for tel_idx, tel_id in enumerate(telescopes['TelID']):
        tel_geom[tel_id] = CameraGeometry.guess(cameras(tel_id)['PixX'].to(u.m),
                                                cameras(tel_id)['PixY'].to(u.m),
                                                telescopes['FL'][tel_idx] * u.m) 


    
    for filename in chain(sorted(filenamelist_gamma)[:], sorted(filenamelist_proton)[:]):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    allowed_tels=range(10),
                                    max_events=args.max_events)

        
        
        
        for event in source:
            mc_shower = event.mc
            mc_shower_core = np.array( [mc_shower.core_x.value, mc_shower.core_y.value] )
            
            
            
            
            tel_data = {}
            tot_signal = 0
            for tel_id in set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data):
                data = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id)
                tel_data[tel_id] = data
                tot_signal += sum(data)
          
            sizes   = []
            widths  = []
            lengths = []
            
            reduced_scaled_width     = []
            reduced_scaled_length    = []
            
            for tel_id, pmt_signal in tel_data.items():

                tel_idx = np.searchsorted( telescopes['TelID'], tel_id )
                tel_pos = np.array( [telescopes["TelX"][tel_idx], telescopes["TelY"][tel_idx]] )
                impact_dist = linalg.length(tel_pos-mc_shower_core)
                
                mask = tailcuts_clean(tel_geom[tel_id], pmt_signal, 1,picture_thresh=10.,boundary_thresh=8.)
                dilate(tel_geom[tel_id], mask)
                
                moments = hillas_parameters(tel_geom[tel_id].pix_x[ mask ],
                                            tel_geom[tel_id].pix_y[ mask ],
                                            pmt_signal[ mask ])
                
                width_mean     = mean_widths    .get_value( [[log10(tot_signal), log10(impact_dist)]] )[0]
                width_sq_mean  = mean_widths_sq .get_value( [[log10(tot_signal), log10(impact_dist)]] )[0]
                length_mean    = mean_lengths   .get_value( [[log10(tot_signal), log10(impact_dist)]] )[0]
                length_sq_mean = mean_lengths_sq.get_value( [[log10(tot_signal), log10(impact_dist)]] )[0]
                
                
                if  width_mean**2  ==  width_sq_mean: continue
                if length_mean**2  == length_sq_mean: continue
                if np.isnan(np.array([width_mean, width_sq_mean, length_mean,
                                      length_sq_mean,moments.length.value,moments.width.value])).any():
                    continue
            
                
                reduced_scaled_width    .append(abs(moments.width .value -  width_mean) /
                                                abs(  width_mean**2  -  width_sq_mean)**.5 )
                reduced_scaled_length   .append(abs(moments.length.value - length_mean) /
                                                abs( length_mean**2  - length_sq_mean)**.5 )
                
                
                sizes  .append(moments.size)
                widths .append(moments.width.value if moments.width.value==moments.width.value else 0)
                lengths.append(moments.length.value)

            if len(sizes) == 0: continue



            sizes     = np.array(sizes)
            size_mean = np.mean(sizes)
            widths     = np.array(widths)
            width_mean = np.mean(widths) 
            size_RMS   = np.mean( ( sizes  -  size_mean )**2 )**.5
            width_RMS  = np.mean( ( widths - width_mean )**2 )**.5
            
            if filename in filenamelist_proton:
                Class = "p"
                mu    = 5.
                sigma = 4.
            else:
                Class = "g"
                mu    = 0.
                sigma = 2.
            
            #Features[Class].append( [size_mean, width_mean, width_RMS, mc_shower.energy.value] )
            Features[Class].append( [log10(size_mean), log10(tot_signal),
                                     gauss(mu,sigma),
                                     gauss(mu,sigma/2.) ])
                                   #sum(reduced_scaled_length)/len(reduced_scaled_length),
                                   #sum(reduced_scaled_width) /len(reduced_scaled_width) ] )

            Classes[Class].append( Class )

            #red_width[Class].append(sum(reduced_scaled_width) /len(reduced_scaled_width) )
            #red_lenth[Class].append(sum(reduced_scaled_length)/len(reduced_scaled_length))
                
            MCEnergy[Class].append(log10(mc_shower.energy.to(u.GeV).value))


        if stop:
            stop = False
            break
    
    lengths = { "p": len(Features["p"]), "g":len(Features["g"]) }
    print("\nfound {} gammas and {} protons\n".format(lengths["g"], lengths["p"]))
    
    
    fig, ax = plt.subplots(4, 8)
    minmax = [ [min([a[j] for a in Features["g"]]+[a[j] for a in Features["p"]]), max([a[j] for a in Features["g"]]+[a[j] for a in Features["p"]])] for j in range(4) ]
    for i in range(4):
        for j in range(4):
            ax[i,  j].hexbin( [a[j] for a in Features["g"]], [a[i] for a in Features["g"]], gridsize=20) 
            ax[i,4+j].hexbin( [a[j] for a in Features["p"]], [a[i] for a in Features["p"]], gridsize=20) 
            ax[i,  j].axis( minmax[j] + minmax[i] )
            ax[i,4+j].axis( minmax[j] + minmax[i] )
    plt.pause(1)


    
    # reduce the number of events so that they are the same in gammas and protons
    NEvents = min(lengths.values())
    for cl in ["p", "g"]:
        Features[cl] = Features[cl][:NEvents]
        Classes [cl] = Classes [cl][:NEvents]
        MCEnergy[cl] = MCEnergy[cl][:NEvents]
    # done

    
    
    
    split_size = 10
    start      = 0
    while start+split_size <= NEvents:
            
        predictFeatures = Features["p"][start:start+split_size] + Features["g"][start:start+split_size]
        predictClasses  = Classes ["p"][start:start+split_size] + Classes ["g"][start:start+split_size]
        predictMCEnergy = MCEnergy["p"][start:start+split_size] + MCEnergy["g"][start:start+split_size]
    
        trainFeatures   = Features["p"][:start] + Features["p"][start+split_size:] + Features["g"][:start] + Features["g"][start+split_size:]
        trainClasses    = Classes ["p"][:start] + Classes ["p"][start+split_size:] + Classes ["g"][:start] + Classes ["g"][start+split_size:]

        start += split_size
        
        
        #clf = svm.SVC(kernel='rbf')
        clf = RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=1, random_state=0)
        clf.fit(trainFeatures, trainClasses)
    
        predict = clf.predict(predictFeatures)
        
        for idx, ev in enumerate(predictClasses):
            total[ev].fill( [predictMCEnergy[idx]] )
            if ev != predict[idx]: wrong[ev].fill( [predictMCEnergy[idx]] )
                    
            
        
        for cl in ["p", "g"]:
            if sum(total[cl].hist) > 0:
                print( "wrong {}: {} out of {} => {}".format(cl, sum(wrong[cl].hist), sum(total[cl].hist),sum(wrong[cl].hist) / sum(total[cl].hist) *100*u.percent))
        print()
        if stop: break

    print()
    print("-"*30)
    print()
    for cl in ["p", "g"]:
        if sum(total[cl].hist) > 0:
            print( "wrong {}: {} out of {} => {}".format(cl, sum(wrong[cl].hist), sum(total[cl].hist),sum(wrong[cl].hist) / sum(total[cl].hist) *100*u.percent))

        wrong[cl].hist[total[cl].hist > 0] = wrong[cl].hist[total[cl].hist > 0] / total[cl].hist[total[cl].hist > 0]
    
    fig = plt.figure()
    plt.subplot(221)
    wrong["g"].draw_1d()
    plt.title("gamma misstag")
    plt.xlabel("log(E/GeV)")
    plt.ylabel("incorrect / all")
    
    plt.subplot(222)
    wrong["p"].draw_1d()
    plt.title("proton misstag")
    plt.xlabel("log(E/GeV)")
    plt.ylabel("incorrect / all")

    plt.subplot(223)
    total["g"].draw_1d()
    plt.title("gamma numbers")
    plt.xlabel("log(E/GeV)")
    plt.ylabel("events")

    plt.subplot(224)
    total["p"].draw_1d()
    plt.title("proton numbers")
    plt.xlabel("log(E/GeV)")
    plt.ylabel("events")

    
    plt.show()
    