from sys import exit, path
from os.path import expandvars
import math
import numpy as np

from glob import glob

from bisect import insort

import matplotlib.pyplot as plt
from matplotlib import cm

from ctapipe.visualization import CameraDisplay

from astropy import units as u
from astropy.table import Table
performance_table = Table(names=("Eps_w", "Eps_t", "diff_Eps",
                                 "alpha_w", "alpha_t",
                                 "hill_width_w", "hill_length_w",
                                 "hill_width_t", "hill_length_t",
                                 "sig_w", "sig_t", "sig_p"))


from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils.linalg import get_phi_theta, set_phi_theta, angle, length

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException


path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe"))

from extract_and_crop_simtel_images import crop_astri_image

from modules.ImageCleaning import ImageCleaner, EdgeEventException
from modules.CutFlow import CutFlow

from helper_functions import *


''' your favourite units here '''
angle_unit  = u.deg
energy_unit = u.GeV
dist_unit   = u.m


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    parser.add_argument('--plot_c',  action='store_true',
                        help="plot camera-wise displays")
    parser.add_argument('--add_offset', action='store_true',
                        help="adds a 15 PE offset to all pixels to supress 'Nbr < 0' "
                             "warnings from mrfilter")

    args = parser.parse_args()

    if args.proton:
        filenamelist = glob("{}/proton/*run{}*gz".format(args.indir, args.runnr))
    else:
        filenamelist = glob("{}/gamma/*run{}*gz".format(args.indir, args.runnr))

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    island_cleaning = True
    Cleaner = {"w": ImageCleaner(mode="wave", cutflow=Imagecutflow,
                                 skip_edge_events=False,
                                 island_cleaning=island_cleaning),
               "t": ImageCleaner(mode="tail", cutflow=Imagecutflow,
                                 skip_edge_events=False,
                                 island_cleaning=island_cleaning)
               }

    fit = FitGammaHillas()

    signal_handler = SignalHandler()
    if args.plot_c:
        signal.signal(signal.SIGINT, signal_handler.stop_drawing)
    else:
        signal.signal(signal.SIGINT, signal_handler)

    NTels = []
    EnMC  = []



    # allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead
    for filename in sorted(filenamelist)[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            Eventcutflow.count("noCuts")

            if len(event.dl0.tels_with_data) < 2:
                continue

            Eventcutflow.count("min2Tels")

            print('Scanning input file... count = {}'.format(event.count))
            print('Event ID: {}'.format(event.dl0.event_id))
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            # getting the MC shower info
            shower = event.mc
            # corsika measures azimuth the other way around, using phi=-az
            shower_dir = set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
            # shower direction is downwards, shower origin up
            shower_org = -shower_dir

            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                '''
                guessing camera geometry '''
                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = 180.*u.deg
                    tel_theta[tel_id] = 20.*u.deg

                '''
                applying ASTRI or general pixel calibration '''
                if cam_geom[tel_id] == "ASTRI":
                    cal_signal = apply_mc_calibration_ASTRI(
                                    event.dl0.tel[tel_id].adc_sums,
                                    event.mc.tel[tel_id].dc_to_pe,
                                    event.mc.tel[tel_id].pedestal)
                else:
                    cal_signal = apply_mc_calibration(
                        event.dl0.tel[tel_id].adc_sums[0],
                        event.mc.tel[tel_id].dc_to_pe[0],
                        event.mc.tel[tel_id].pedestal[0])

                Imagecutflow.count("calibration")

                '''
                now cleaning the image with wavelet and tail cuts '''
                try:
                    pmt_signal_w, new_geom_w = \
                        Cleaner['w'].clean(cal_signal+5 if args.add_offset else cal_signal,
                                           cam_geom[tel_id], event.inst.optical_foclen[tel_id])
                    pmt_signal_t, new_geom_t = \
                        Cleaner['t'].clean_tail(cal_signal.copy(), cam_geom[tel_id],
                                                event.inst.optical_foclen[tel_id])
                except (FileNotFoundError, EdgeEventException) as e:
                    print(e)
                    continue

                '''
                do some plotting '''
                if args.plot_c and signal_handler.draw:
                    fig = plt.figure()

                    ax1 = fig.add_subplot(221)
                    disp1 = CameraDisplay(cam_geom[tel_id],
                                          image=event.mc.tel[tel_id].photo_electron_image,
                                          ax=ax1)
                    disp1.cmap = plt.cm.hot
                    disp1.add_colorbar()
                    plt.title("PE image")

                    ax2 = fig.add_subplot(222)
                    disp2 = CameraDisplay(cam_geom[tel_id],
                                          image=cal_signal,
                                          ax=ax2)
                    disp2.cmap = plt.cm.hot
                    disp2.add_colorbar()
                    plt.title("calibrated noisy image")

                    ax3 = fig.add_subplot(223)
                    disp3 = CameraDisplay(new_geom_t,
                                          image=pmt_signal_t,
                                          ax=ax3)
                    disp3.cmap = plt.cm.hot
                    disp3.add_colorbar()
                    plt.title("tail cleaned")

                    ax4 = fig.add_subplot(224)
                    disp4 = CameraDisplay(new_geom_w,
                                          image=np.sum(pmt_signal_w, axis=1)
                                          if pmt_signal_w.shape[-1] == 25 else pmt_signal_w,
                                          ax=ax4)
                    disp4.cmap = plt.cm.hot
                    disp4.add_colorbar()
                    plt.title("wave cleaned")
                    plt.suptitle("Camera {}".format(tel_id))
                    plt.show()

                '''
                do the hillas parametrisation of the two cleaned images '''
                try:
                    hillas = {}
                    hillas['w'] = hillas_parameters(new_geom_w.pix_x,
                                                    new_geom_w.pix_y,
                                                    pmt_signal_w)[0]
                    hillas['t'] = hillas_parameters(new_geom_t.pix_x,
                                                    new_geom_t.pix_y,
                                                    pmt_signal_t)[0]
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

                ''' get some more parameters and put them in a astropy.table.Table '''
                alpha = {}
                length = {}
                width = {}
                for k in Cleaner.keys():
                    fit.get_great_circles({tel_id: hillas[k]},
                                          event.inst, tel_phi, tel_theta)
                    c = fit.circles[tel_id]
                    h = hillas[k]
                    alpha[k] = abs((angle(c.norm, shower_org)*u.rad) - 90*u.deg).to(u.deg)
                    length[k] = h.length * u.m
                    width[k] = h.width * u.m

                sum_w = np.sum(pmt_signal_w)
                sum_p = np.sum(event.mc.tel[tel_id].photo_electron_image)
                Epsilon_intensity_w = abs(sum_w - sum_p) / sum_p

                sum_t = np.sum(pmt_signal_t)
                Epsilon_intensity_t = abs(sum_t - sum_p) / sum_p
                performance_table.add_row([Epsilon_intensity_w, Epsilon_intensity_t,
                                           Epsilon_intensity_w - Epsilon_intensity_t,
                                           alpha['w'], alpha['t'],
                                           width['w'], length['w'],
                                           width['t'], length['t'],
                                           sum_w, sum_t, sum_p])

            '''
            determine and print the 68-percentile of the hillas ellipsis tilt error
            of the two cleaning methods '''
            alphas_w = performance_table["alpha_w"]
            alphas_t = performance_table["alpha_t"]
            print()
            print("alpha_w res (68-percentile) = {}".format(np.percentile(alphas_w, 68)))
            print("alpha_t res (68-percentile) = {}".format(np.percentile(alphas_t, 68)))

            if signal_handler.stop: break
        if signal_handler.stop: break

    '''
    print the cutflows for telescopes and camera images '''
    print()
    Eventcutflow("min2Tels")
    print()
    Imagecutflow(sort_column=1)

    print(performance_table)

    if args.write:
        performance_table.write("Eps_int_comparison.fits", overwrite=True)

    '''
    if we don't want to plot anything, we can exit now '''
    if not args.plot:
        exit(0)

    npe_edges = np.linspace(1, 6, 21)
    size_edges = npe_edges
    lovw_edges = np.linspace(0, 3, 16)

    sig_p = performance_table["sig_p"]

    for k in Cleaner.keys():

        Epsilon_2 = performance_table["Eps_{}".format(k)]
        hillas_tilt = performance_table["alpha_{}".format(k)] * u.deg
        hillas_length = performance_table["hill_length_{}".format(k)]
        hillas_width = performance_table["hill_width_{}".format(k)]
        sig = performance_table["sig_{}".format(k)]

        mode = "wave" if k == "w" else "tail"

        plot_hex_and_violin(
                np.log10(Epsilon_2),
                np.log10(sig_p),
                bin_edges=None,
                ylabel="log10(NPE)",
                xlabel="log10(Epsilon 2)",
                zlabel="log10(counts)",
                bins='log',
                extent=(-3, 0, 1.5, 5),
                do_violin=False)
        plt.suptitle(mode)
        plt.pause(.1)

        '''
        plot the angular error of the hillas ellipsis vs the number of photo electrons '''
        plot_hex_and_violin(np.log10(sig_p),
                            np.log10(hillas_tilt/angle_unit),
                            npe_edges,
                            extent=[0, 5, -5, 1],
                            xlabel="log10(number of photo electrons)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        if args.write:
            save_fig("plots/alpha_vs_photoelecrons_{}".format(mode))
        plt.pause(.1)

        '''
        plot the angular error of the hillas
        ellipsis vs the measured signal on the camera '''
        plot_hex_and_violin(np.log10(sig_p),
                            np.log10(hillas_tilt/angle_unit),
                            size_edges,
                            extent=[0, 5, -5, 1],
                            xlabel="log10(signal size)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        if args.write:
            save_fig("plots/alpha_vs_signal_{}".format(mode))
        plt.pause(.1)

        '''
        plot the angular error of the hillas ellipsis vs the length/width ratio '''
        plot_hex_and_violin(np.log10(hillas_length/hillas_width),
                            np.log10(hillas_tilt/angle_unit),
                            lovw_edges,
                            extent=[0, 2, -4.5, 1],
                            xlabel="log10(length/width)",
                            ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
        plt.suptitle(mode)
        if args.write:
            save_fig("plots/alpha_vs_lenOVwidth_{}".format(mode))
        plt.pause(.1)


    plt.show()

