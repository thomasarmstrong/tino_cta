from sys import exit, path
from os.path import expandvars
import math
import numpy as np

from glob import glob

import matplotlib.pyplot as plt
from matplotlib import cm

from astropy import units as u

from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.containers import MCShowerData as MCShower

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


old = False
# old = True


''' your favourite units here '''
angle_unit  = u.deg
energy_unit = u.GeV
dist_unit   = u.m


if __name__ == '__main__':

    args = make_argparser().parse_args()

    filenamelist = glob("{}/gamma/*run{}*gz".format(args.indir, args.runnr))
    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    # Cleaner = ImageCleaner(mode="none")
    Cleaner = ImageCleaner(mode=args.mode)

    fit = FitGammaHillas()
    fit.setup_geometry(*load_hessio(filenamelist[0]),
                       phi=180*u.deg, theta=20*u.deg)

    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    NTels = []
    EnMC  = []
    xis1  = []
    xis2  = []
    xisb  = []

    diffs = []

    tel_geom = {}

    allowed_tels = range(10)  # smallest 3×3 square of ASTRI telescopes
    # allowed_tels = range(34)  # all ASTRI telescopes
    for filename in sorted(filenamelist):
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
            print('available telscopes: {}'.format(event.dl0.tels_with_data))

            hillas_dict = {}
            # for tel_id, tel in event.mc.tel.items():
                # pmt_signal = tel.photo_electrons

            for tel_id in event.dl0.tels_with_data:

                Imagecutflow.count("noCuts")

                pmt_signal = apply_mc_calibration_ASTRI(
                    event.dl0.tel[tel_id].adc_sums, tel_id)

                Imagecutflow.count("calibration")

                if tel_id not in tel_geom:
                    tel_geom[tel_id] = CameraGeometry.guess(
                                        fit.cameras(tel_id)['PixX'].to(u.m),
                                        fit.cameras(tel_id)['PixY'].to(u.m),
                                        fit.telescopes['FL'][tel_id-1] * u.m)

                try:
                    pmt_signal, pix_x, pix_y = \
                        Cleaner.clean(pmt_signal, tel_geom[tel_id])
                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEventException:
                    continue

                Imagecutflow.count("cleaned")

                if np.sum(pmt_signal) < args.min_charge:
                    continue

                Imagecutflow.count("minCharge")

                try:
                    hillas_dict[tel_id] = hillas_parameters(pix_x, pix_y, pmt_signal)[0]
                except HillasParameterizationError as e:
                    print(e)
                    continue

                Imagecutflow.count("Hillas")

            Eventcutflow.count("ImagePreparation")

            fit.get_great_circles(hillas_dict)

            Eventcutflow.count("GreatCircles")

            shower = event.mc
            # corsika measures azimuth the other way around, using phi=-az
            shower_dir = set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
            # shower direction is downwards, shower origin up
            shower_org = -shower_dir

            shower_core = np.array([shower.core_x.value,
                                    shower.core_y.value])*u.m

            try:
                result1, crossings = fit.fit_origin_crosses()
                result2            = result1
                result2            = fit.fit_origin_minimise(result1)

                seed = np.sum([[fit.telescopes["TelX"][tel_id-1],
                                fit.telescopes["TelY"][tel_id-1]]
                              for tel_id in fit.circles.keys()], axis=0) * u.m
                pos_fit = fit.fit_core(seed)
            except TooFewTelescopesException as e:
                print(e)
                continue

            Eventcutflow.count("Reco")

            xi1 = angle(result1, shower_org).to(angle_unit)
            xi2 = angle(result2, shower_org).to(angle_unit)

            print()
            print("xi1 = {}".format(xi1))
            print("xi2 = {}".format(xi2))
            print("x1-xi2 = {}".format(xi1-xi2))
            xis1.append(xi1)
            xis2.append(xi2)
            # xisb.append(min(xi1, xi2))

            NEvents = len(xis2)
            print()
            print("xi1 res (68-percentile) = {}"
                  .format(sorted(xis1)[int(NEvents*.68)]))
            print("xi2 res (68-percentile) = {}"
                  .format(sorted(xis2)[int(NEvents*.68)]))
            print()

            diff = length(pos_fit[:2]-shower_core)
            print("reco = ", diff)
            diffs.append(diff)
            print("core res (68-percentile) = {}"
                  .format(sorted(diffs)[int(len(diffs)*.68)]))
            print()
            print("Events:", NEvents)
            print()

            '''
            save number of telescopes and MC energy for this event '''
            NTels.append(len(fit.circles))
            EnMC.append(event.mc.energy)

            if signal_handler.stop: break
        if signal_handler.stop: break

    '''
    print the cutflows for telescopes and camera images '''
    print()
    Eventcutflow()
    print()
    Imagecutflow()

    '''
    if we don't want to plot anything, we can exit now '''
    if not args.plot:
        exit(0)

    # xis1 = convert_astropy_array(xis1)
    xis2 = convert_astropy_array(xis2)
    # xisb = convert_astropy_array(xisb)

    xis = xis2

    figure = plt.figure()
    plt.hist(xis, bins=np.linspace(0, 25, 50), log=True)
    plt.xlabel(r"$\xi_1$ / deg")
    if args.write:
        tikz_save('plots/'+args.mode+'_xi1.tex', draw_rectangles=True)
        plt.savefig('plots/'+args.mode+'_xi1.png')
        plt.savefig('plots/'+args.mode+'_xi1.pdf')
    plt.pause(.1)

    #figure = plt.figure()
    #plt.hist(np.log10(xis2/angle_unit), bins=np.linspace(-3, 1, 50))
    #plt.xlabel(r"log($\xi_2$ / deg)")
    #if args.write:
        #tikz_save('plots/'+args.mode+'_xi2.tex', draw_rectangles=True)
        #plt.savefig('plots/'+args.mode+'_xi2.png')
        #plt.savefig('plots/'+args.mode+'_xi2.pdf')
    #plt.pause(.1)

    #figure = plt.figure()
    #plt.hist(np.log10(xisb/angle_unit), bins=np.linspace(-.5, .5, 50))
    #plt.xlabel(r"$\log(\xi_\mathrm{best} / \deg)$")
    #if args.write:
        #tikz_save('plots/'+args.mode+'_xi_best.tex', draw_rectangles=True)
        #plt.savefig('plots/'+args.mode+'_xi_best.png')
        #plt.savefig('plots/'+args.mode+'_xi_best.pdf')
    #plt.pause(.1)

    #figure = plt.figure()
    #plt.hist((xis1-xis2), bins=np.linspace(-.65, .65, 13), log=True)
    #plt.xlabel(r"$(\xi_1 - \xi_2) / \deg)$")
    #if args.write:
        #tikz_save('plots/'+args.mode+'_xi_diff.tex', draw_rectangles=True)
        #plt.savefig('plots/'+args.mode+'_xi_diff.png')
        #plt.savefig('plots/'+args.mode+'_xi_diff.pdf')
    #plt.pause(.1)

    '''
    convert the xi-list into a dict with the number of
    used telescopes as keys '''
    xi_vs_tel = {}
    for xi, ntel in zip(xis, NTels):
        if math.isnan(xi.value):
            continue
        if ntel not in xi_vs_tel:
            xi_vs_tel[ntel] = [xi/angle_unit]
        else:
            xi_vs_tel[ntel].append(xi/angle_unit)

    '''
    create a list of energy bin-edges and -centres for violine plots '''
    Energy_edges = np.linspace(2, 8, 13)
    Energy_centres = (Energy_edges[1:]+Energy_edges[:-1])/2.

    '''
    convert the xi-list in to an energy-binned dict with
    the bin centre as keys '''
    xi_vs_energy = {}
    for en, xi in zip(EnMC, xis):
        if math.isnan(xi.value):
            continue
        ''' get the bin number this event belongs into '''
        ebin = np.digitize(np.log10(en/energy_unit), Energy_edges)-1
        ''' the central value of the bin is the key for the dictionary '''
        if Energy_centres[ebin] not in xi_vs_energy:
            xi_vs_energy[Energy_centres[ebin]]  = [xi/angle_unit]
        else:
            xi_vs_energy[Energy_centres[ebin]] += [xi/angle_unit]


    '''
    plotting the angular error as violine plots with binning in
    number of telescopes an shower energy '''
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in xi_vs_tel.values()],
                   [a for a in xi_vs_tel.keys()],
                   points=60, widths=.75, showextrema=True, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in xi_vs_energy.values()],
                   [a for a in xi_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=True, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\xi_2$ / deg)")
    plt.grid()
    if args.write:
        tikz_save('plots/'+args.mode+'_xi_vs_E_NTel.tex')
        plt.savefig('plots/'+args.mode+'_xi_vs_E_NTel.png')
        plt.savefig('plots/'+args.mode+'_xi_vs_E_NTel.pdf')

    plt.pause(.1)

    '''
    convert the diffs-list into a dict with the number of
    used telescopes as keys '''
    diff_vs_tel = {}
    for diff, ntel in zip(diffs, NTels):
        if ntel not in diff_vs_tel:
            diff_vs_tel[ntel] = [(diff/dist_unit)]
        else:
            diff_vs_tel[ntel].append((diff/dist_unit))

    '''
    convert the diffs-list in to an energy-binned dict with
    the bin centre as keys '''
    diff_vs_energy = {}
    for en, diff in zip(EnMC, diffs):
        ''' get the bin number this event belongs into '''
        ebin = np.digitize(np.log10(en/energy_unit), Energy_edges)-1
        ''' the central value of the bin is the key for the dictionary '''
        if Energy_centres[ebin] not in diff_vs_energy:
            diff_vs_energy[Energy_centres[ebin]]  = [(diff/dist_unit)]
        else:
            diff_vs_energy[Energy_centres[ebin]] += [(diff/dist_unit)]

    '''
    plotting the core position error as violine plots with binning in
    number of telescopes an shower energy '''
    figure = plt.figure()
    plt.subplot(211)
    plt.violinplot([np.log10(a) for a in diff_vs_tel.values()],
                   [a for a in diff_vs_tel.keys()],
                   points=60, widths=.75, showextrema=True, showmedians=True)
    plt.xlabel("Number of Telescopes")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()

    plt.subplot(212)
    plt.violinplot([np.log10(a) for a in diff_vs_energy.values()],
                   [a for a in diff_vs_energy.keys()],
                   points=60, widths=(Energy_edges[1]-Energy_edges[0])/1.5,
                   showextrema=True, showmedians=True)
    plt.xlabel(r"log(Energy / GeV)")
    plt.ylabel(r"log($\Delta R$ / m)")
    plt.grid()
    if args.write:
        tikz_save('plots/'+args.mode+'_dist_vs_E_NTel.tex')
        plt.savefig('plots/'+args.mode+'_dist_vs_E_NTel.png')
        plt.savefig('plots/'+args.mode+'_dist_vs_E_NTel.pdf')
    plt.show()
