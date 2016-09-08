from sys import exit, path

path.append("/local/home/tmichael/software/jeremie_cta/snippets/ctapipe")
from extract_and_crop_simtel_images import crop_astri_image

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")
from datapipe.denoising.wavelets_mrtransform import wavelet_transform


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
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave")
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


    mean_widths     = Histogram(initFromFITS="tail_widths.fits")
    mean_widths_sq  = Histogram(initFromFITS="tail_widths_sq.fits")
    mean_lengths    = Histogram(initFromFITS="tail_lengths.fits")
    mean_lengths_sq = Histogram(initFromFITS="tail_lengths_sq.fits")

    axisNames = [ "log(E / GeV)" ]
    ranges    = [ [2,8] ]
    nbins     = [ 7 ]
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

        
        if filename in filenamelist_proton:
            Class = "p"
        else:
            Class = "g"

        
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
            dists   = []
            
            wsum  = 0
            lsum  = 0
            wsum2 = 0
            lsum2 = 0
            sumWeigthWidth = 0
            sumLengthWidth = 0
            
            scaled_width  = []
            scaled_length = []
            weight_width  = []
            weight_length = []
            
            for tel_id, pmt_signal in tel_data.items():

                tel_idx = np.searchsorted( telescopes['TelID'], tel_id )
                tel_pos = np.array( [telescopes["TelX"][tel_idx], telescopes["TelY"][tel_idx]] )
                impact_dist = linalg.length(tel_pos-mc_shower_core)
                dists.append(impact_dist)
                
                
                if args.mode == "wave":
                    # for now wavelet library works only on rectangular images
                    cropped_img = crop_astri_image(pmt_signal)
                    pmt_signal = wavelet_transform(cropped_img, 4, "/tmp/wavelet").flatten()
                    
                    # hillas parameter function requires image and x/y arrays to be of the same dimension
                    pix_x = crop_astri_image(tel_geom[tel_id].pix_x).flatten()
                    pix_y = crop_astri_image(tel_geom[tel_id].pix_y).flatten()
        
                elif args.mode == "tail":
                    mask = tailcuts_clean(tel_geom[tel_id], pmt_signal, 1,picture_thresh=10.,boundary_thresh=8.)
                    if True not in mask: continue
                    dilate(tel_geom[tel_id], mask)
                    pmt_signal[mask] = 0
                    pix_x = tel_geom[tel_id].pix_x
                    pix_y = tel_geom[tel_id].pix_y
                else: 
                    raise Exception('cleaning mode "{}" not found'.format(mode))
                
                moments, h_moments = hillas_parameters(pix_x, pix_y, pmt_signal)
                signal = moments.size
                width_mean     = mean_widths    .get_value( [[log10(signal), log10(impact_dist)]] )[0]
                width_sq_mean  = mean_widths_sq .get_value( [[log10(signal), log10(impact_dist)]] )[0]
                length_mean    = mean_lengths   .get_value( [[log10(signal), log10(impact_dist)]] )[0]
                length_sq_mean = mean_lengths_sq.get_value( [[log10(signal), log10(impact_dist)]] )[0]
                
                
                # sometimes weird things happen
                if  width_mean**2  ==  width_sq_mean: continue
                if length_mean**2  == length_sq_mean: continue
                if np.isnan(np.array([width_mean, width_sq_mean, length_mean,
                                      length_sq_mean,moments.length,moments.width])).any():
                                    continue
                

                
                scaled_width .append((moments.width  -  width_mean) / ( width_sq_mean -  width_mean**2)**.5)
                weight_width .append(1)
                scaled_length.append((moments.length - length_mean) / (length_sq_mean - length_mean**2)**.5)
                weight_length.append(1)
                
                
                sizes  .append(moments.size)
                widths .append(moments.width)
                lengths.append(moments.length)


            if len(sizes) == 0: continue

            
            scaled_width = np.array(scaled_width) 
            weight_width = np.array(weight_width)
            scaled_length = np.array(scaled_length) 
            weight_length = np.array(weight_length)
            wsum = sum( scaled_width  * weight_width ) / sum(weight_width)
            lsum = sum( scaled_length * weight_length) / sum(weight_length)
            sigmaW = sum(weight_width )**.5
            sigmaL = sum(weight_length)**.5

            sizes       = np.array(sizes)
            size_mean   = np.mean(sizes)
            #widths      = np.array(widths)
            #width_mean  = np.mean(widths) 
            #lengths     = np.array(lengths)
            #length_mean = np.mean(lengths) 
            #size_RMS    = np.mean( ( sizes  -  size_mean )**2 )**.5
            #width_RMS   = np.mean( ( widths  - width_mean  )**2 )**.5
            #length_RMS  = np.mean( ( lengths - length_mean )**2 )**.5
            
            
            #Features[Class].append( [size_mean, width_mean, length_mean, width_RMS, length_RMS, size_RMS, sum(sizes)] )
            Features[Class].append( [log10(size_mean), log10(tot_signal),
                                   wsum , lsum] )

            Classes[Class].append( Class )

            red_width[Class].append(wsum)
            red_lenth[Class].append(lsum)
                
            MCEnergy[Class].append(log10(mc_shower.energy.to(u.GeV).value))


        if stop:
            stop = False
            break
    
    lengths = { "p": len(Features["p"]), "g":len(Features["g"]) }
    print("\nfound {} gammas and {} protons\n".format(lengths["g"], lengths["p"]))
    
    ''' creating 2D plots of all features '''
    NFeatures = len(Features["p"][0])
    fig, ax = plt.subplots(4,2*NFeatures)
    minmax = [ [min([a[j] for a in Features["g"]]+[a[j] for a in Features["p"]]), max([a[j] for a in Features["g"]]+[a[j] for a in Features["p"]])] for j in range(4) ]
    
    for i in range(NFeatures):
        for j in range(NFeatures):
            #for i, foo in enumerate(a):
                #for j, bar in enumerate(b):
                    ax1 = ax[i,j]
                    ax2 = ax[i,j+NFeatures]
                    if i == j:
                        ax1.hist( [a[j] for a in Features["g"]] )
                        ax2.hist( [a[j] for a in Features["p"]] )
                        #ax1.yscale('log', nonposy='clip')
                        #ax2.yscale('log', nonposy='clip')
                    else:
                        ax1.hexbin( [a[j] for a in Features["g"]], [a[i] for a in Features["g"]], bins='log', gridsize=20) 
                        ax2.hexbin( [a[j] for a in Features["p"]], [a[i] for a in Features["p"]], bins='log', gridsize=20) 
                        ax1.axis( minmax[j] + minmax[i] )
                        ax2.axis( minmax[j] + minmax[i] )
            #break
        #break
    plt.pause(1)


    
    # reduce the number of events so that they are the same in gammas and protons
    NEvents = min(lengths.values())
    for cl in ["p", "g"]:
        Features[cl] = Features[cl][:NEvents]
        Classes [cl] = Classes [cl][:NEvents]
        MCEnergy[cl] = MCEnergy[cl][:NEvents]


    
    
    
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
    
    
    plt.style.use('seaborn-talk')
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
    