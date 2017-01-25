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
from ctapipe.image.geometry_converter import *


path.append(expandvars("$CTA_SOFT/tino_cta"))
from modules.ImageCleaning import kill_isolpix
from helper_functions import *


global tel_geom

wavelet_transform = WaveletTransform()

fig = None
from ctapipe.visualization import CameraDisplay
def transform_and_clean_hex_image(signal, cam_geom, optical_foclen, photo_electrons):
    rot_x, rot_y = unskew_hex_pixel_grid(cam_geom.pix_x, cam_geom.pix_y,
                                         cam_geom.cam_rotation)
    colors = cm.hot(signal/max(signal))

    x_edges, y_edges, x_scale = get_orthogonal_grid_edges(rot_x, rot_y)

    new_geom, new_signal = convert_geometry_1d_to_2d(
        cam_geom, pmt_signal, cam_geom.cam_id)

    unrot_geom, unrot_signal = convert_geometry_back(
        new_geom, new_signal, cam_geom.cam_id,
        event.inst.optical_foclen[tel_id])

    square_mask = new_geom.mask

    cleaned_img = wavelet_transform(new_signal,
                                    raw_option_string="-K -C1 -m1 -f2 -s3 -n4")
    cleaned_img = kill_isolpix(cleaned_img)

    unrot_img = cleaned_img[square_mask == 1]
    unrot_colors = cm.hot(unrot_img/max(unrot_img))

    global fig
    global cb1, ax1
    global cb2, ax2
    global cb3, ax3
    global cb4, ax4
    global cb5, ax5
    global cb6, ax6
    if fig is None:
        fig = plt.figure()
    else:
        fig.delaxes(ax1)
        fig.delaxes(ax2)
        fig.delaxes(ax3)
        fig.delaxes(ax4)
        fig.delaxes(ax5)
        fig.delaxes(ax6)
        cb1.remove()
        cb2.remove()
        cb3.remove()
        cb5.remove()
        cb6.remove()
    ax1 = fig.add_subplot(231)
    disp1 = CameraDisplay(cam_geom, image=pmt_signal, ax=ax1)
    disp1.cmap = plt.cm.hot
    disp1.add_colorbar()
    cb1 = disp1.colorbar
    plt.title("noisy image")

    ax2 = fig.add_subplot(232)
    plt.hist2d(rot_x, rot_y, bins=(x_edges, y_edges), cmap=cm.hot,
               weights=signal)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("noisy, slanted image")
    cb2 = plt.colorbar()

    ax3 = fig.add_subplot(233)
    plt.imshow(np.sqrt(cleaned_img.T), interpolation='none', cmap=cm.hot, origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, slanted image")
    cb3 = plt.colorbar()
    ax3.set_axis_off()

    ax4 = fig.add_subplot(234)
    plt.imshow(square_mask.T, interpolation='none', cmap=cm.hot, origin='lower')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("mask for slanted image")
    ax4.set_axis_off()

    ax5 = fig.add_subplot(235)
    disp5 = CameraDisplay(cam_geom, image=photo_electrons, ax=ax5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("photo-electron image")
    disp5.add_colorbar()
    cb5 = disp5.colorbar

    ax6 = fig.add_subplot(236)
    disp6 = CameraDisplay(unrot_geom, image=np.sqrt(unrot_img),
                          ax=ax6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("cleaned, original geometry")
    disp6.add_colorbar()
    cb6 = disp6.colorbar

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

    filenamelist_gamma  = glob("{}/gamma/run{}.*gz".format(args.indir, args.runnr))
    filenamelist_proton = glob("{}/proton/run{}.*gz".format(args.indir, args.runnr))

    print("{}/gamma/run{}.*gz".format(args.indir, args.runnr))
    if len(filenamelist_gamma) == 0:
        print("no gammas found...")
    if len(filenamelist_proton) == 0:
        print("no protons found...")

    cam_geom = {}

    allowed_tels = range(34, 39)
    # allowed_tels=range(10), # smallest ASTRI aray
    # allowed_tels=range(34), # all ASTRI telescopes

    for filename in chain(sorted(filenamelist_gamma)[:], sorted(filenamelist_proton)[:]):
        filename = expandvars(
                         "$HOME/Data/cta/Prod3/gamma_20deg_180deg_run1000___cta-prod3"
                         "-demo_desert-2150m-Paranal-demo2sect_cone10.simtel.gz")
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     #allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            pmt_signal = None

            for tel_id in event.dl0.tels_with_data:

                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                if cam_geom[tel_id].pix_type is not "hexagonal":
                    continue

                if False:
                    pmt_signal = apply_mc_calibration_ASTRI(
                        event.dl0.tel[tel_id].adc_sums, tel_id)
                else:
                    pmt_signal = apply_mc_calibration(
                        event.dl0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0])

                if False:
                    ''' TODO switch to cleaning module if still desired '''
                else:

                    transform_and_clean_hex_image(pmt_signal,
                                                  cam_geom[tel_id],
                                                  event.inst.optical_foclen[tel_id],
                                                  event.mc.tel[tel_id].photo_electron_image)
                    continue

