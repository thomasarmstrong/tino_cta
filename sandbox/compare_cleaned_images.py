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


path.append(expandvars("$CTA_SOFT/tino_cta"))
from modules.ImageCleaning import kill_isolpix

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


def apply_mc_calibration(adcs, gains, peds):
    """
    apply basic calibration
    """

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


def unskew_hex_pixel_grid(pix_x, pix_y, angle=60*u.deg):
    """
        transform the pixel coordinates of a hexagonal image into an orthogonal image

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the hexagonal pixel grid
        angle : astropy.Quantity (default: 60 degrees)
            the skewing angle of the hex-grid. should be 60° for regular hexagons

        Returns:
        --------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid
    """

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rot_x = pix_x*(cos_angle + sin_angle)
    rot_y = pix_y*sin_angle - pix_x*cos_angle

    return rot_x, rot_y


def reskew_hex_pixel_grid(pix_x, pix_y, angle=60*u.deg):
    """
        skews the orthogonal coordinates back to the hexagonal ones

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid
        angle : astropy.Quantity (default: 60 degrees)
            the skewing angle of the hex-grid. should be 60° for regular hexagons

        Returns:
        --------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the hexagonal pixel grid
    """

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    rot_x = pix_x / (cos_angle+sin_angle)
    rot_y = pix_x*cos_angle/(sin_angle*cos_angle + sin_angle**2) + pix_y/sin_angle

    return rot_x, rot_y


def reskew_hex_pixel_from_orthogonal_edges(x_edges, y_edges, square_mask):
    """
        extracts and skews the pixel coordinates from a 2D orthogonal histogram
        (i.e. the bin-edges) and skews them into the hexagonal image while selecting only
        the pixel that are selected by the given mask

        Parameters:
        -----------
        x_edges, y_edges : 1darrays
            the bin edges of the 2D histogram
        square_mask : 2darray
            mask that selects the pixels actually belonging to the camera

        Returns:
        --------
        unrot_x, unrot_y : 1darrays
            pixel coordinated reskewed into the hexagonal camera grid
    """

    unrot_x, unrot_y = [], []
    for i, x in enumerate((x_edges[:-1]+x_edges[1:])/2):
        for j, y in enumerate((y_edges[:-1]+y_edges[1:])/2):
            if square_mask[i][j]:
                x_unrot, y_unrot = reskew_hex_pixel_grid(x, y)
                unrot_x.append(x_unrot)
                unrot_y.append(y_unrot)
    return unrot_x, unrot_y


def get_orthogonal_grid_edges(pix_x, pix_y):
    """
        calculate the bin edges of the slanted, orthogonal pixel grid to resample the
        pixel signals with np.histogramdd right after.

        Parameters:
        -----------
        pix_x, pix_y : 1D numpy arrays
            the list of x and y coordinates of the slanted, orthogonal pixel grid

        Returns:
        --------
        x_edges, y_edges : 1D numpy arrays
            the bin edges for the slanted, orthogonal pixel grid
    """

    '''
    finding the size of the square patches '''
    d_x = 99 * u.m
    d_y = 99 * u.m
    x_base = pix_x[0]
    y_base = pix_y[0]
    for x, y in zip(pix_x, pix_y):
        if x == x_base: continue
        if abs(y-y_base) < abs(x-x_base):
            d_x = min(d_x, abs(x-x_base))
    for x, y in zip(pix_x, pix_y):
        if y == y_base: continue
        if abs(y-y_base) > abs(x-x_base):
            d_y = min(d_y, abs(y-y_base))

    '''
    with the maximal extension of the axes and the size of the pixels, determine the
    number of bins in each direction '''
    NBinsx = np.around(abs(max(pix_x) - min(pix_x))/d_x) + 2
    NBinsy = np.around(abs(max(pix_y) - min(pix_y))/d_y) + 2
    x_edges = np.linspace(min(pix_x).value, max(pix_x).value, NBinsx)
    y_edges = np.linspace(min(pix_y).value, max(pix_y).value, NBinsy)

    return x_edges, y_edges

fig = None
def transform_and_crop_hex_image(signal, pix_x, pix_y):
    rot_x, rot_y = unskew_hex_pixel_grid(pix_x, pix_y)
    colors = cm.hot(signal/max(signal))

    x_edges, y_edges = get_orthogonal_grid_edges(rot_x, rot_y)
    square_img, edges = np.histogramdd([rot_x, rot_y],
                                       bins=(x_edges, y_edges),
                                       weights=signal)

    square_mask, edges = np.histogramdd([rot_x, rot_y],
                                        bins=(x_edges, y_edges))

    cleaned_img = wavelet_transform(square_img)
    cleaned_img = kill_isolpix(cleaned_img)

    unrot_img = cleaned_img[square_mask == 1].flatten()
    unrot_colors = cm.hot(unrot_img/max(unrot_img))

    unrot_x, unrot_y = reskew_hex_pixel_from_orthogonal_edges(x_edges,
                                                              y_edges,
                                                              square_mask)

    global fig
    global cb1
    global cb2
    if fig is None:
        fig = plt.figure()
    else:
        cb1.remove()
        cb2.remove()
    fig.add_subplot(221)
    plt.scatter(pix_x, pix_y, color=colors, marker='H', s=25)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("noisy image")

    fig.add_subplot(222)
    plt.hist2d(rot_x, rot_y, bins=(x_edges, y_edges), cmap=cm.hot,
               weights=signal, vmin=0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("noisy, slanted image")
    cb1 = plt.colorbar()

    ax = fig.add_subplot(223)
    plt.imshow(cleaned_img.T, interpolation='none', cmap=cm.hot, origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, slanted image")
    cb2 = plt.colorbar()
    ax.set_axis_off()

    fig.add_subplot(224)
    plt.scatter(unrot_x, unrot_y, color=unrot_colors, marker='H', s=25)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, original geometry")

    plt.pause(.1)
    response = input("press return to continue")
    if response != "":
        exit()



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
                    data = apply_mc_calibration_ASTRI(
                        event.dl0.tel[tel_id].adc_sums, tel_id)
                else:
                    data = apply_mc_calibration(
                        event.dl0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0] )
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



