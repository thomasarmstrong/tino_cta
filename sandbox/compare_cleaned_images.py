from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe"))

from glob import glob
import argparse

from itertools import chain

import numpy as np

from astropy import units as u

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

from extract_and_crop_simtel_images import crop_astri_image
from datapipe.denoising.wavelets_mrfilter import WaveletTransform

from ctapipe.io import CameraGeometry
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.io.hessio import hessio_event_source

global tel_geom

wavelet_transform = WaveletTransform()


import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id, adc_tresh=3500):
    """
    apply basic calibration
    """
    gains = pyhessio.get_calibration(tel_id)

    calibrated = [ (adc0-971)*gain0 if adc0 < adc_tresh else (adc1-961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains[0],gains[1]) ]
    return np.array(calibrated)


def get_mc_calibration_coeffs(tel_id):
    """
    Get the calibration coefficients from the MC data file to the
    data.  This is ahack (until we have a real data structure for the
    calibrated data), it should move into `ctapipe.io.hessio_event_source`.

    returns
    -------
    (peds,gains) : arrays of the pedestal and pe/dc ratios.
    """
    peds = pyhessio.get_pedestal(tel_id)[0]
    gains = pyhessio.get_calibration(tel_id)[0]
    return peds, gains


def apply_mc_calibration(adcs, tel_id):
    """
    apply basic calibration
    """
    peds, gains = get_mc_calibration_coeffs(tel_id)

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


def transform_and_crop_hex_image(signal,pix_x, pix_y,angle=60*u.deg):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rot_x = pix_x*cos_angle + pix_x*sin_angle
    rot_y = pix_y*sin_angle - pix_x*cos_angle
    colors = cm.hot(signal/max(signal))

    max_x = max( rot_x[signal>0] )
    min_x = min( rot_x[signal>0] )
    max_y = max( rot_y[signal>0] )
    min_y = min( rot_y[signal>0] )

    cropped_x = rot_x [(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_y = rot_y [(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_s = signal[(rot_x<max_x) & (rot_x > min_x) & (rot_y < max_y) & (rot_y > min_y)]
    cropped_c = cm.hot(cropped_s/max(cropped_s))



    ''' finding the size of the square patches '''
    d_x = 99 * u.m
    d_y = 99 * u.m
    x_base = cropped_x[0]
    y_base = cropped_y[0]
    for x, y in zip(cropped_x, cropped_y):
        if x == x_base: continue
        if abs(y-y_base) < abs(x-x_base):
            d_x = min(d_x, abs(x-x_base))
    for x, y in zip(cropped_x, cropped_y):
        if y == y_base: continue
        if abs(y-y_base) > abs(x-x_base):
            d_y = min(d_y, abs(y-y_base))

    NBinsx = np.around(abs(max(cropped_x) - min(cropped_x))/d_x) + 2
    NBinsy = np.around(abs(max(cropped_y) - min(cropped_y))/d_y) + 2
    x_edges = np.linspace(min(cropped_x), max(cropped_x), NBinsx)
    y_edges = np.linspace(min(cropped_y), max(cropped_y), NBinsy)
    square_img, edges = np.histogramdd([cropped_y, cropped_x],
                                       bins=(y_edges, x_edges), weights=cropped_s )

    img_mean = np.mean(cropped_s)
    for i, line in enumerate(square_img):
        for j, v in enumerate(line):
            if v == 0:
                square_img[i][j] = np.random.poisson(img_mean)


    cleaned_img = wavelet_transform(square_img)

    fig = plt.figure()
    fig.add_subplot(221)
    plt.scatter(pix_x, pix_y,color=colors,marker='H', s=25)
    plt.gca().set_aspect('equal', adjustable='box')
    fig.add_subplot(222)
    plt.scatter(rot_x, rot_y,color=colors,marker='s', s=21)
    plt.gca().set_aspect('equal', adjustable='box')

    ax3 = fig.add_subplot(223)
    plt.imshow(cleaned_img, interpolation='none', cmap=cm.hot, origin='lower')
    #plt.hist2d(cropped_x, cropped_y, bins=(x_edges, y_edges), weights=cropped_s,
               #cmap=cm.hot)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str,
                        default=expandvars("$HOME/Data/cta/ASTRI9"))
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
                    data = apply_mc_calibration(event.dl0.tel[tel_id].adc_sums[0], tel_id)
                    #data = event.dl0.tel[tel_id].adc_sums[0]
                tel_data[tel_id] = data


            for tel_id, pmt_signal in tel_data.items():

                if False:
                    ''' TODO switch to cleaning module if still desired '''
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



