from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline/"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe/"))

from itertools import chain

from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import linalg
from ctapipe.instrument.InstrumentDescription import load_hessio
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError

from helper_functions import *
from modules.CutFlow import CutFlow
from modules.ImageCleaning import ImageCleaner, \
                                  EdgeEventException, UnknownModeException


from datapipe.classifiers.EventClassifier \
        import EventClassifier

from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException









if __name__ == '__main__':

    parser = make_argparser()
    args = parser.parse_args()

    filenamelist_gamma  = glob("{}/gamma/run{}.*gz".format(args.indir, args.runnr))
    filenamelist_proton = glob("{}/proton/run{}.*gz".format(args.indir, args.runnr))

    print("{}/gamma/run{}.*gz".format(args.indir, args.runnr))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    '''
    prepare InstrumentDescription '''
    InstrDesc = load_hessio(filenamelist_gamma[0])

    '''
    wrapper for the scikit learn classifier '''
    classifier = EventClassifier()
    classifier.setup_geometry(*InstrDesc,
                              phi=180*u.deg, theta=20*u.deg)
    classifier.cleaner = ImageCleaner(args.mode)
    classifier.load("data/classify_pickle/classifier_" +
                    args.mode + "_rec-sim-dist.pkl")

    '''
    simple hillas-based shower reco '''
    fit = FitGammaHillas()
    fit.setup_geometry(*InstrDesc,
                       phi=180*u.deg, theta=20*u.deg)

    '''
    class that wraps tail cuts and wavelet cleaning for ASTRI telescopes '''
    Cleaner = ImageCleaner(mode=args.mode)

    '''
    to have geometry information accessible here as well '''
    tel_geom = classifier.tel_geom

    '''
    catch ctr-c signal to exit current loop and still display results '''
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    agree_threshold = .75
    min_tel = 2



    source_orig = None
    fit_origs   = {'g':[], 'p':[]}
    MC_Energy   = {'g':[], 'p':[]}
    multiplicity = {'g':[], 'p':[]}

    events_total         = {'g':0, 'p':0}
    events_passd_telcut1 = {'g':0, 'p':0}
    events_passd_telcut2 = {'g':0, 'p':0}
    events_passd_gsel    = {'g':0, 'p':0}
    telescopes_total     = {'g':0, 'p':0}
    telescopes_passd     = {'g':0, 'p':0}

    for filenamelist_class in [filenamelist_gamma,
                               filenamelist_proton]:

        cl = "g" if "gamma" in filenamelist_class[0] else "p"

        for filename in sorted(filenamelist_class)[:args.last]:
            print("filename = {}".format(filename))

            source = hessio_event_source(
                filename,
                allowed_tels=range(10),  # smallest ASTRI aray
                # allowed_tels=range(34),  # all ASTRI telescopes
                max_events=args.max_events)

            for event in source:
                events_total[cl] += 1

                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                if source_orig is None:
                    '''
                    corsika measures azimuth the other way around,
                    using phi=-az '''
                    source_dir = linalg.set_phi_theta(-mc_shower.az,
                                                      90.*u.deg+mc_shower.alt)
                    '''
                    shower direction is downwards, shower origin up '''
                    source_orig = -source_dir

                NTels = len(event.dl0.tels_with_data)
                '''
                skip events with less than minimum hit telescopes '''
                if NTels < min_tel:
                    continue
                events_passd_telcut1[cl] += 1

                '''
                telescope loop '''
                tot_signal = 0
                hillas_dict1 = {}
                hillas_dict2 = {}
                for tel_id in event.dl0.tels_with_data:
                    classifier.total_images += 1

                    pmt_signal = apply_mc_calibration_ASTRI(
                                event.dl0.tel[tel_id].adc_sums, tel_id)
                    '''
                    trying to clean the image '''
                    try:
                        pmt_signal, pix_x, pix_y = \
                            Cleaner.clean(pmt_signal, tel_geom[tel_id])
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    except EdgeEventException:
                        continue
                    except UnknownModeException as e:
                        print(e)
                        print("asked for unknown mode... what are you doing?")
                        exit(-1)

                    '''
                    trying to do the hillas reconstruction of the images '''
                    try:
                        moments, h_moments = hillas_parameters(pix_x, pix_y,
                                                               pmt_signal)

                        hillas_dict1[tel_id] = moments
                        hillas_dict2[tel_id] = h_moments
                        tot_signal += moments.size

                    except HillasParameterizationError as e:
                        print(e)
                        print("ignoring this camera")
                        pass

                '''
                telescope loop done, now do the core fit '''
                fit.get_great_circles(hillas_dict1)
                seed = np.sum([[fit.telescopes["TelX"][tel_id-1],
                                fit.telescopes["TelY"][tel_id-1]]
                        for tel_id in fit.circles.keys()], axis=0) * u.m
                pos_fit = fit.fit_core(seed)

                '''
                now prepare the features for the classifier '''
                features = []
                NTels = len(hillas_dict1)
                for tel_id in hillas_dict1.keys():
                    tel_idx = np.searchsorted(
                                classifier.telescopes['TelID'],
                                tel_id)
                    tel_pos = np.array([
                        classifier.telescopes["TelX"][tel_idx],
                        classifier.telescopes["TelY"][tel_idx]
                                        ]) * u.m

                    moments = hillas_dict1[tel_id]
                    h_moments = hillas_dict2[tel_id]

                    impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                    impact_dist_rec = linalg.length(tel_pos-pos_fit)
                    features.append([
                                impact_dist_rec / u.m,
                                impact_dist_sim / u.m,
                                tot_signal,
                                moments.size,
                                NTels,
                                moments.width, moments.length,
                                h_moments.Skewness,
                                h_moments.Kurtosis,
                                h_moments.Asymmetry
                                ])

                telescopes_total[cl] += len(set(event.trig.tels_with_trigger) &
                                            set(event.dl0.tels_with_data))
                telescopes_passd[cl] += len(features)

                if len(features) == 0:
                    continue

                try:
                    predict = classifier.predict(
                        [tel[:1]+tel[2:] for tel in features])
                except Exception as e:
                    print("error: ", e)
                    print("Ntels: {}, Nfeatures: {}".format(
                        len(set(event.trig.tels_with_trigger) &
                            set(event.dl0.tels_with_data)),
                        len(features)))
                    print("skipping event")
                    continue

                isGamma = [1 if (tel == "g") else 0 for tel in predict]

                '''
                skip events with less than minimum hit telescopes
                where the event is not on the edge '''
                if len(isGamma) < min_tel: continue
                events_passd_telcut2[cl] += 1
                '''
                skip events where too few classifiers agree it's a gamma '''
                if np.mean(isGamma) <= agree_threshold: continue
                events_passd_gsel[cl] += 1

                '''
                reconstruct direction now '''
                # fit.get_great_circles(hillas_dict1)
                result1, crossings = fit.fit_origin_crosses()
                result2 = result1

                fit_origs[cl].append(result2)
                MC_Energy[cl].append(event.mc.energy/u.GeV)
                multiplicity[cl].append(NTels)

            if signal_handler.stop:
                stop = False
                break

    off_angles = {'p': [], 'g': []}
    phi = {'g': [], 'p': []}
    the = {'g': [], 'p': []}
    for cl, in fit_origs.keys():
        for fit in fit_origs[cl]:
            off_angles[cl].append(linalg.angle(fit, source_orig)/u.deg)
            phithe = linalg.get_phi_theta(fit)
            phi[cl].append((phithe[0] if phithe[0] > 0
                            else phithe[0]+360*u.deg)/u.deg)
            the[cl].append(phithe[1]/u.deg)
        off_angles[cl] = np.array(off_angles[cl])

    if args.write:
        from astropy.table import Table
        for cl in ['g', 'p']:
            Table([off_angles[cl], MC_Energy[cl], phi[cl], the[cl], NTels[cl]],
                  names=("off_angles", "MC_Energy", "phi", "theta", "multiplicity")
                  ).write("data/selected_events/selected_events_" +
                          args.mode+"_"+cl+".fits",
                          overwrite=True)

    for cl in ['g', 'p']:
        print(cl)
        print("telescopes_total: {}, telescopes_passd: {}, passed/total: {}"
              .format(telescopes_total[cl], telescopes_passd[cl],
                      telescopes_passd[cl]/telescopes_total[cl]))
        print("events_total: {},\n"
              "events_passd_telcut1: {}, passed/total telcut1: {},\n"
              "events_passd_telcut2: {}, passed/total telcut2: {},\n"
              "events_passd_gsel: {}, passed/total gsel: {} \n"
              "passd gsel / passd telcut: {}"
              .format(events_total[cl],
                      events_passd_telcut1[cl],
                      events_passd_telcut1[cl]/events_total[cl]
                      if events_total[cl] > 0 else 0,
                      events_passd_telcut2[cl],
                      events_passd_telcut2[cl]/events_total[cl]
                      if events_total[cl] > 0 else 0,
                      events_passd_gsel[cl],
                      events_passd_gsel[cl]/events_total[cl]
                      if events_total[cl] > 0 else 0,
                      events_passd_gsel[cl]/events_passd_telcut2[cl]
                      if events_passd_telcut2[cl] > 0 else 0))
        print()

    print("selected {} gammas and {} proton events"
          .format(len(fit_origs['g']), len(fit_origs['p'])))

    weight_g = 1
    weight_p = 1e5

    if args.plot:
        fig = plt.figure()
        plt.subplot(311)
        plt.hist([off_angles['p'], off_angles['g']],
                 weights=[[weight_p]*len(off_angles['p']),
                          [weight_g]*len(off_angles['g'])],
                 rwidth=1, bins=50, stacked=True)
        plt.xlabel("alpha")

        plt.subplot(312)
        plt.hist([off_angles['p']**2, off_angles['g']**2],
                 weights=[[weight_p]*len(off_angles['p']),
                          [weight_g]*len(off_angles['g'])],
                 rwidth=1, bins=50, stacked=True)
        plt.xlabel("alpha²")

        plt.subplot(313)
        plt.hist([-np.cos(off_angles['p']), -np.cos(off_angles['g'])],
                 weights=[[weight_p]*len(off_angles['p']),
                          [weight_g]*len(off_angles['g'])],
                 rwidth=1, bins=50, stacked=True)
        plt.xlabel("-cos(alpha)")
        plt.pause(.1)

        fig2 = plt.figure()
        unit = u.deg
        plt.hist2d(convert_astropy_array(chain(phi['p'], phi['g']), unit),
                   convert_astropy_array(chain(the['p'], the['g']), unit),
                   range=([[(180-3), (180+3)],
                           [(20-3), (20+3)]]*u.deg).to(unit).value)
        plt.xlabel("phi / {}".format(unit))
        plt.ylabel("theta / {}".format(unit))
        plt.show()
