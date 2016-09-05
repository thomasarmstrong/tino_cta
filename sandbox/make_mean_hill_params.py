from sys import exit, path

path.append("/local/home/tmichael/software/jeremie_cta/snippets/ctapipe")
from extract_and_crop_simtel_images import crop_astri_image

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")
from datapipe.denoising.wavelets_mrtransform import wavelet_transform

from glob import glob
import argparse

from ctapipe.reco.hillas import hillas_parameters
from ctapipe.reco.cleaning import tailcuts_clean, dilate

from math import log10

import numpy as np

from astropy import units as u

from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils import linalg
from ctapipe.utils.fitshistogram import Histogram


import matplotlib.pyplot as plt



import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id):
    """
    apply basic calibration
    """
    peds0 = pyhessio.get_pedestal(tel_id)[0]
    peds1 = pyhessio.get_pedestal(tel_id)[1]
    gains0 = pyhessio.get_calibration(tel_id)[0]
    gains1 = pyhessio.get_calibration(tel_id)[1]
    
    calibrated = [ (adc0-971)*gain0 if adc0 < 3500 else (adc1-961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains0,gains1) ]
    return np.array(calibrated)


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


    tel_geom = {}
    for tel_idx, tel_id in enumerate(telescopes['TelID']):
        tel_geom[tel_id] = CameraGeometry.guess(cameras(tel_id)['PixX'].to(u.m),
                                                cameras(tel_id)['PixY'].to(u.m),
                                                telescopes['FL'][tel_idx] * u.m)
    axisNames  = ["log(SignalSize)","log(ImpactDist)"]
    nbins      = [12,12]
    ranges     = [[3,6],[0.5,3.5]]
    max_dist   = 0
    min_sign   = 1e9
    max_sign   = 0
    widths     = Histogram(axisNames=axisNames,nbins=nbins,ranges=ranges,name="widths")
    widths_sq  = Histogram(axisNames=axisNames,nbins=nbins,ranges=ranges,name="widths_squared")
    lengths    = Histogram(axisNames=axisNames,nbins=nbins,ranges=ranges,name="lengths")
    lengths_sq = Histogram(axisNames=axisNames,nbins=nbins,ranges=ranges,name="lengths_squared")
    norms      = Histogram(axisNames=axisNames,nbins=nbins,ranges=ranges,name="norms")

    signal.signal(signal.SIGINT, signal_handler)

    #for filename in sorted(filenamelist_gamma)[:]:
    for filename in sorted(filenamelist_proton)[:]:
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    allowed_tels=range(33),
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
            max_sign = max(max_sign,tot_signal)
            min_sign = min(min_sign,tot_signal)
          
            for tel_id, pmt_signal in tel_data.items():

                tel_idx = np.searchsorted( telescopes['TelID'], tel_id )
                tel_pos = np.array( [telescopes["TelX"][tel_idx], telescopes["TelY"][tel_idx]] )
                impact_dist = linalg.length(tel_pos-mc_shower_core)
                
                max_dist = max(max_dist,impact_dist)
                
                #mask = tailcuts_clean(tel_geom[tel_id], pmt_signal, 1,picture_thresh=10.,boundary_thresh=8.)
                #if True not in mask: continue
                #dilate(tel_geom[tel_id], mask)
                #pmt_signal[mask==False] = 0
                
                pix_x = crop_astri_image(tel_geom[tel_id].pix_x).flatten()
                pix_y = crop_astri_image(tel_geom[tel_id].pix_y).flatten()
                cropped_img = crop_astri_image(pmt_signal)
                pmt_signal = wavelet_transform(cropped_img, 4, "/tmp/wavelet").flatten()
        
                
                moments = hillas_parameters(pix_x, pix_y, pmt_signal)
                
                if moments.width != moments.width:   continue
                if moments.length != moments.length: continue
            
                widths    .fill([[np.clip(log10(tot_signal),3,6)], [np.clip(log10(impact_dist),0.5,3.5)]], weights=[moments.width.value])
                widths_sq .fill([[np.clip(log10(tot_signal),3,6)], [np.clip(log10(impact_dist),0.5,3.5)]], weights=[moments.width.value**2])
                lengths   .fill([[np.clip(log10(tot_signal),3,6)], [np.clip(log10(impact_dist),0.5,3.5)]], weights=[moments.length.value])
                lengths_sq.fill([[np.clip(log10(tot_signal),3,6)], [np.clip(log10(impact_dist),0.5,3.5)]], weights=[moments.length.value**2])
                norms     .fill([[np.clip(log10(tot_signal),3,6)], [np.clip(log10(impact_dist),0.5,3.5)]])

                if stop: break
            if stop: break
        print("maximal distance is:",max_dist)
        print("minimal   signal is:",min_sign)
        print("maximal   signal is:",max_sign)
        if stop: break


    widths    .hist[norms.hist > 0] = widths    .hist[norms.hist > 0] / norms.hist[norms.hist > 0]
    widths_sq .hist[norms.hist > 0] = widths_sq .hist[norms.hist > 0] / norms.hist[norms.hist > 0]
    lengths   .hist[norms.hist > 0] = lengths   .hist[norms.hist > 0] / norms.hist[norms.hist > 0]
    lengths_sq.hist[norms.hist > 0] = lengths_sq.hist[norms.hist > 0] / norms.hist[norms.hist > 0]
    
    
    
    fig = plt.figure()

    plt.subplot(2,3,1)
    widths .draw_2d()
    plt.colorbar()

    plt.subplot(2,3,2)
    lengths.draw_2d()
    plt.colorbar()    
        
    plt.subplot(2,3,4)
    widths_sq .draw_2d()
    plt.colorbar()

    plt.subplot(2,3,5)
    lengths_sq.draw_2d()
    plt.colorbar()

    plt.subplot(2,3,3)
    norms.draw_2d()
    plt.colorbar()

    plt.show()
    
    
    widths    .to_fits().writeto(    "wave_widths.fits",clobber=True)
    widths_sq .to_fits().writeto( "wave_widths_sq.fits",clobber=True)
    lengths   .to_fits().writeto(   "wave_lengths.fits",clobber=True)
    lengths_sq.to_fits().writeto("wave_lengths_sq.fits",clobber=True)
    