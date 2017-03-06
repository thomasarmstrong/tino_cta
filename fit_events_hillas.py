from sys import exit, path
from os.path import expandvars
import math
import numpy as np

from glob import glob

from bisect import insort

import matplotlib.pyplot as plt
from matplotlib import cm

from astropy import units as u
from astropy.table import Table

from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils import linalg

from ctapipe.image.hillas import HillasParameterizationError, \
                                 hillas_parameters_4 as hillas_parameters

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
    parser.add_argument('-o', '--outdir',   type=str, default="plots")
    parser.add_argument('--photon',  action='store_true',
                        help="use the mc photo-electrons container "
                        "instead of the PMT signal")
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")

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

    Eventcutflow.set_cut("noCuts", None)
    Eventcutflow.set_cut("min2Tels", lambda x: x < 2)
    Eventcutflow.set_cut("min2Images", lambda x: x < 2)
    Eventcutflow.set_cut("GreatCircles", None)
    Eventcutflow.set_cut("nan pos", lambda x: np.isnan(x.value).any())
    Eventcutflow.set_cut("nan dir", lambda x: np.isnan(x).any())

    min_charge = "min charge >= {}".format(args.min_charge)
    Imagecutflow.set_cut(min_charge, lambda x: x >= args.min_charge)

    Cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=False, island_cleaning=True)

    fit = FitGammaHillas()

    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    reco_table = Table(names=("NTels", "EnMC", "xi", "DR"),
                       dtype=('i', 'f', 'f', 'f'))
    reco_table["EnMC"].unit = energy_unit
    reco_table["xi"].unit = angle_unit
    reco_table["DR"].unit = dist_unit

    tel_signal = []
    tel_signal_pe = []
    hillas_tilt = []
    hillas_length = []
    hillas_width = []

    allowed_tels = None
    allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    # allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = range(34, 40)  # use the array of FlashCams instead

    for filename in sorted(filenamelist)[:args.last]:

        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            Eventcutflow.count("noCuts")

            if Eventcutflow.cut("min2Tels", len(event.dl0.tels_with_data)):
                continue

            print('Scanning input file... count = {}'.format(event.count))
            print('Event ID: {}'.format(event.dl0.event_id))
            print('Available telscopes: {}'.format(event.dl0.tels_with_data))

            hillas_dict = {}
            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                    tel_theta[tel_id] = (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad

                if args.photon:
                    pmt_signal = event.mc.tel[tel_id].photo_electron_image
                    new_geom = cam_geom[tel_id]
                else:
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

                    try:
                        pmt_signal, new_geom = \
                            Cleaner.clean(cal_signal, cam_geom[tel_id],
                                          event.inst.optical_foclen[tel_id])
                    except (FileNotFoundError, EdgeEventException) as e:
                        continue
                # end if args.photons

                if not Imagecutflow.keep(min_charge, np.sum(pmt_signal)):
                    continue

                try:
                    h = hillas_parameters(new_geom.pix_x,
                                          new_geom.pix_y,
                                          pmt_signal)
                    if h.length > 0 and h.width > 0:
                        hillas_dict[tel_id] = h
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

            if Eventcutflow.cut("min2Images", len(hillas_dict)):
                continue

            fit.get_great_circles(hillas_dict, event.inst, *tel_orientation,
                                  cam_rotation=cam_geom[tel_id].cam_rotation)

            Eventcutflow.count("GreatCircles")

            shower = event.mc
            shower_org = linalg.set_phi_theta(shower.az, 90.*u.deg-shower.alt)

            for k in fit.circles.keys():
                c = fit.circles[k]
                h = hillas_dict[k]

                tel_signal.append(h.size)
                tel_signal_pe.append(np.sum(event.mc.tel[k].photo_electron_image))
                hillas_tilt.append(abs(linalg.angle(c.norm, shower_org)*u.rad - 90*u.deg))
                hillas_length.append(h.length)
                hillas_width.append(h.width)

            shower_core = convert_astropy_array([shower.core_x, shower.core_y])

            try:
                fit_position = fit.fit_core_crosses()
            except Exception as e:
                print([c.norm for c in fit.circles.values()])
                raise e
            if Eventcutflow.cut("nan pos", fit_position):
                continue

            fit_origin = fit.fit_origin_crosses()[0]
            if Eventcutflow.cut("nan dir", fit_origin):
                continue

            print("fit_position:", fit_position)
            print("fit_origin:", fit_origin)

            xi = linalg.angle(fit_origin, shower_org).to(angle_unit)
            diff = linalg.length(fit_position[:2]-shower_core)

            reco_table.add_row([len(fit.circles), event.mc.energy.to(energy_unit),
                                xi.to(angle_unit), diff.to(dist_unit)])

            print()
            print("xi = {:4.3f}".format(xi))
            print("xi res (68-percentile) = {:4.3f} {}"
                  .format(np.percentile(reco_table["xi"], 68), angle_unit))

            print()
            print("reco = {:4.3f}".format(diff))
            print("core res (68-percentile) = {:4.3f} {}"
                  .format(np.percentile(reco_table["DR"], 68), dist_unit))
            print()

            if signal_handler.stop: break
        if signal_handler.stop: break

    '''
    print the cutflows for telescopes and camera images '''
    print()
    Eventcutflow("min2Tels")
    print()
    Imagecutflow(sort_column=1)

    '''
    if we don't want to plot anything, we can exit now '''
    if not args.plot:
        exit(0)

    hillas_tilt = convert_astropy_array(hillas_tilt)
    hillas_length = convert_astropy_array(hillas_length)
    hillas_width = convert_astropy_array(hillas_width)

    '''
    plot the angular error of the hillas ellipsis vs the number of photo electrons '''
    npe_edges = np.linspace(1, 6, 21)
    plot_hex_and_violin(np.log10(tel_signal_pe),
                        np.log10(hillas_tilt/angle_unit),
                        npe_edges,
                        extent=[0, 5, -5, 1],
                        xlabel="log10(number of photo electrons)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)
    if args.store:
        from os.path import expandvars
        reco_table.write("rec_events_{}.hdf5".format(args.mode), format="hdf5",
            path=expandvars("$PWD/data/reconstructed_events/rec_events_{}.hdf5".format(args.mode)))

    plt.tight_layout()
    if args.write:
        save_fig("{}/reco_alpha_vs_photoelecrons_{}".format(args.outdir, args.mode))
    plt.pause(.1)

    '''
    plot the angular error of the hillas ellipsis vs the measured signal on the camera '''
    size_edges = npe_edges
    plot_hex_and_violin(np.log10(tel_signal),
                        np.log10(hillas_tilt/angle_unit),
                        size_edges,
                        extent=[0, 5, -5, 1],
                        xlabel="log10(signal size)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)

    plt.tight_layout()
    if args.write:
        save_fig("{}/reco_alpha_vs_signal_{}".format(args.outdir, args.mode))
    plt.pause(.1)

    '''
    plot the angular error of the hillas ellipsis vs the length/width ratio '''
    lovw_edges = np.linspace(0, 1.5, 16)
    plot_hex_and_violin(np.log10(hillas_length/hillas_width),
                        np.log10(hillas_tilt/angle_unit),
                        lovw_edges,
                        extent=[0, 1.5, -4.5, 1],
                        xlabel="log10(length/width)",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)
    plt.tight_layout()

    lovw_edges = np.linspace(0, 1, 16)
    plot_hex_and_violin(1-hillas_width/hillas_length,
                        np.log10(hillas_tilt/angle_unit),
                        lovw_edges,
                        extent=[0, 1, -4.5, 1],
                        xlabel="1-width/length",
                        ylabel=r"log10($\alpha$/{:latex})".format(angle_unit))
    plt.suptitle(args.mode)

    plt.tight_layout()
    if args.write:
        save_fig("{}/reco_alpha_vs_eccentricity_{}".format(args.outdir, args.mode))
    plt.pause(.1)

    xis = reco_table["xi"]

    figure = plt.figure()
    plt.hist(xis, bins=np.linspace(0, .1, 50), log=True)
    plt.xlabel(r"$\xi$ / deg")
    if args.write:
        save_fig('{}/reco_xi_{}'.format(args.outdir, args.mode), draw_rectangles=True)
    plt.pause(.1)

    # convert the xi-list into a dict with the number of used telescopes as keys
    xi_vs_tel = {}
    for xi, ntel in reco_table["xi", "NTels"]:
        if ntel not in xi_vs_tel:
            xi_vs_tel[ntel] = [xi]
        else:
            xi_vs_tel[ntel].append(xi)

    # create a list of energy bin-edges and -centres for violin plots
    Energy_edges = np.linspace(2, 8, 13)
    Energy_centres = (Energy_edges[1:]+Energy_edges[:-1])/2.

    # convert the xi-list in to an energy-binned dict with the bin centre as keys
    xi_vs_energy = {}
    for en, xi in reco_table["EnMC", "xi"]:

        # get the bin number this event belongs into
        sbin = np.digitize(np.log10(en), Energy_edges)-1

        # the central value of the bin is the key for the dictionary
        if Energy_centres[sbin] not in xi_vs_energy:
            xi_vs_energy[Energy_centres[sbin]]  = [xi]
        else:
            xi_vs_energy[Energy_centres[sbin]] += [xi]

    # plotting the angular error as violin plots with binning in
    # number of telescopes and shower energy
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in xi_vs_tel.values()],
                   [a for a in xi_vs_tel.keys()],
                   points=60, widths=.75, showextrema=False, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in xi_vs_energy.values()],
                   [a for a in xi_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=False, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()

    plt.tight_layout()
    if args.write:
        save_fig('{}/reco_xi_vs_E_NTel_{}'.format(args.outdir, args.mode))

    plt.pause(.1)

    # convert the diffs-list into a dict with the number of used telescopes as keys
    diff_vs_tel = {}
    for diff, ntel in reco_table["DR", "NTels"]:
        if ntel not in diff_vs_tel:
            diff_vs_tel[ntel] = [diff]
        else:
            diff_vs_tel[ntel].append(diff)

    '''
    convert the diffs-list in to an energy-binned dict with
    the bin centre as keys '''
    diff_vs_energy = {}
    for en, diff in reco_table["EnMC", "DR"]:

        # get the bin number this event belongs into
        sbin = np.digitize(np.log10(en), Energy_edges)-1

        # the central value of the bin is the key for the dictionary
        if Energy_centres[sbin] not in diff_vs_energy:
            diff_vs_energy[Energy_centres[sbin]]  = [diff]
        else:
            diff_vs_energy[Energy_centres[sbin]] += [diff]

    # plotting the core position error as violin plots with binning in
    # number of telescopes an shower energy
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in diff_vs_tel.values()],
                   [a for a in diff_vs_tel.keys()],
                   points=60, widths=.75, showextrema=False, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in diff_vs_energy.values()],
                   [a for a in diff_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=False, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.tight_layout()
    if args.write:
        save_fig('{}/reco_dist_vs_E_NTel_{}'.format(args.outdir, args.mode))
    plt.show()
