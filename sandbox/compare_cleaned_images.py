from sys import exit, path

path.append("/local/home/tmichael/software/jeremie_cta/snippets/ctapipe")

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")


from glob import glob
import argparse

from itertools import chain

import numpy as np

from astropy import units as u

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from extract_and_crop_simtel_images import crop_astri_image
from datapipe.denoising.wavelets_mrtransform import wavelet_transform

from ctapipe.io import CameraGeometry
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.io.hessio import hessio_event_source
from ctapipe.reco.cleaning import tailcuts_clean, dilate

from random import random
wave_out_name = "/tmp/wavelet_{}_".format(random())
global tel_geom

import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id, adc_tresh=3500):
    """
    apply basic calibration
    """
    gains = pyhessio.get_calibration(tel_id)
    
    calibrated = [ (adc0-971)*gain0 if adc0 < adc_tresh else (adc1-961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains[0],gains[1]) ]
    return np.array(calibrated)



def transform_and_crop_hex_image(signal,pix_x, pix_y,angle=60*u.deg):
    rot_x = pix_x*np.cos(angle) - pix_y*np.sin(angle)
    rot_y = pix_y*np.sin(angle) + pix_y*np.cos(angle)
    colors = cm.hot(signal/max(signal))
    
    max_x = max( rot_x[signal>0] )
    min_x = min( rot_x[signal>0] )
    max_y = max( rot_y[signal>0] )
    min_y = min( rot_y[signal>0] )
    
    cropped_x = rot_x [(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_y = rot_y [(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_s = signal[(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_c = cm.hot(cropped_s/max(cropped_s))
    
    x_max = cropped_x[cropped_s == max(cropped_s)]
    min_min_y = min(cropped_y[ np.abs(cropped_x-x_max[0])<.03*u.m ])
    
    print(x_max, min_min_y)
    
    cropped_cropped_x = cropped_x[cropped_y>min_min_y]
    cropped_cropped_y = cropped_y[cropped_y>min_min_y]
    cropped_cropped_s = cropped_s[cropped_y>min_min_y]
    cropped_cropped_c = cm.hot(cropped_cropped_s/max(cropped_cropped_s))
    
    fig = plt.figure()
    plt.subplot(221)
    plt.scatter(pix_x, pix_y,color=colors,marker='H',s=20)
    plt.subplot(222)
    plt.scatter(rot_x, rot_y,color=colors,marker='s',s=5)
    plt.subplot(223)
    plt.scatter(cropped_x, cropped_y,color=cropped_c,marker='s',s=5)
    plt.subplot(224)
    plt.scatter(cropped_cropped_x, cropped_cropped_y,color=cropped_cropped_c,marker='s',s=5)
    plt.show()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/Data/cta/ASTRI9")
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('-t', '--teltype', type=str, default="SST_ASTRI")
    parser.add_argument('--tail', dest="mode", action='store_true')
    args = parser.parse_args()
    
    filenamelist_gamma  = glob( "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    filenamelist_proton = glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))
    
    print(  "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    if len(filenamelist_gamma) == 0:
        print("no gammas found...")
    if len(filenamelist_proton) == 0:
        print("no protons found...")
    
    tel_geom = None
    
    allowed_tels=range(34,39)
    #allowed_tels=range(10), # smallest ASTRI aray
    #allowed_tels=range(34), # all ASTRI telescopes
    
    for filename in chain(sorted(filenamelist_gamma)[:], sorted(filenamelist_proton)[:]):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    allowed_tels=allowed_tels,
                                    max_events=args.max_events)
        if tel_geom == None:
            (h_telescopes, h_cameras, h_optics) = load_hessio(filename)
            Ver = 'Feb2016'
            TelVer = 'TelescopeTable_Version{}'.format(Ver)
            CamVer = 'CameraTable_Version{}_TelID'.format(Ver)
            OptVer = 'OpticsTable_Version{}_TelID'.format(Ver)
            
            telescopes = h_telescopes[TelVer]
            cameras    = lambda tel_id : h_cameras[CamVer+str(tel_id)]
            optics     = lambda tel_id : h_optics [OptVer+str(tel_id)]

            for tel_idx, tel_id in enumerate(telescopes['TelID']):
                if tel_id not in allowed_tels:continue
                tel_geom = CameraGeometry.guess(cameras(tel_id)['PixX'].to(u.m),
                                                cameras(tel_id)['PixY'].to(u.m),
                                                telescopes['FL'][tel_idx] * u.m) 
                print(tel_id, tel_geom.pix_type)
                break
            
    
        for event in source:
            tel_data = {}
            for tel_id in event.dl0.tels_with_data:
                if False:
                    data = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id)
                else:
                    data = event.mc.tel[tel_id].photo_electrons
                tel_data[tel_id] = data
            
            
            for tel_id, pmt_signal in tel_data.items():
                
                if False:
                    # for now wavelet library works only on rectangular images
                    cropped_img = crop_astri_image(pmt_signal)
                    cleaned_wave = wavelet_transform(cropped_img, 4, wave_out_name) #.flatten()
                    cleaned_wave -= np.mean(cleaned_wave)
                    cleaned_wave[cleaned_wave<0] = 0
                    
                    
                    # hillas parameter function requires image and x/y arrays to be of the same dimension
                    cropped_pix_x = crop_astri_image(tel_geom.pix_x).flatten()
                    cropped_pix_y = crop_astri_image(tel_geom.pix_y).flatten()

                    # tail cleaning
                    mask = tailcuts_clean(tel_geom, pmt_signal, 1,picture_thresh=10.,boundary_thresh=8.)
                    if True not in mask: continue
                    dilate(tel_geom, mask)
                    cleaned_tail = pmt_signal[:]
                    cleaned_tail[mask==False] = 0
                    cleaned_tail = crop_astri_image(cleaned_tail)
                    
                    cropped_mask = crop_astri_image(mask)
                else:
                    
                    transform_and_crop_hex_image(pmt_signal, tel_geom.pix_x, tel_geom.pix_y)
                    continue
                    
                fig = plt.figure()
                plt.subplot(231)
                plt.imshow(cropped_img,interpolation='none')
                plt.colorbar()
                
                plt.subplot(232)
                plt.imshow(cleaned_wave,interpolation='none')
                plt.colorbar()
                
                plt.subplot(233)
                plt.imshow(cleaned_tail,interpolation='none')                
                plt.colorbar()
                
                plt.subplot(236)
                plt.imshow(cropped_mask,interpolation='none')
                plt.colorbar()

                plt.show()
                
                
                
