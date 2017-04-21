#!/usr/bin/env python3

from helper_functions import *

from sys import exit, path
from os.path import expandvars
from glob import glob
from itertools import chain

from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException

from ctapipe.utils import linalg

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from modules.CutFlow import *
from modules.ImageCleaning import *
from modules.EnergyRegressor import *


def get_class_string(cls):
    print(cls)
    return str(cls.__class__).split('.')[-1].split("'")[0]


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--outdir', type=str,
                        default='data/classifier_pickle')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    args = parser.parse_args()

    filenamelist_gamma  = glob("{}/gamma/run*gz".format(args.indir))

    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # counting events and where they might have gone missing
    Eventcutflow = CutFlow("EventCutFlow Protons")
    Eventcutflow.set_cut("noCuts", None)
    Eventcutflow.set_cut("min2Tels trig", lambda x: x < 2)
    Eventcutflow.set_cut("min2Tels reco", lambda x: x < 2)

    Imagecutflow = CutFlow("ImageCutFlow Protons")
    Imagecutflow.set_cut("position nan", lambda x: np.isnan(x).any())
    Imagecutflow.set_cut("features nan", lambda x: np.isnan(x).any())

    # class that wraps tail cuts and wavelet cleaning
    Cleaner = ImageCleaner(mode=args.mode, wavelet_options=args.raw,
                           skip_edge_events=False)  # args.skip_edge_events)

    # simple hillas-based shower reco
    fit = FitGammaHillas()

    Features_event_list = []
    MC_Energies = []


    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    allowed_tels = np.arange(10).tolist() + np.arange(34,41).tolist()
    for filename in filenamelist_gamma[:4][:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # event loop
        for event in source:

                Eventcutflow.count("noCuts")

                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                if Eventcutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                    continue

                # telescope loop
                tot_signal = 0
                max_signal = 0
                hillas_dict = {}
                for tel_id in event.dl0.tels_with_data:
                    Imagecutflow.count("noCuts")

                    # guessing camera geometry
                    if tel_id not in cam_geom:
                        cam_geom[tel_id] = CameraGeometry.guess(
                                            event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            event.inst.optical_foclen[tel_id])
                        tel_phi[tel_id] = 0.*u.deg
                        tel_theta[tel_id] = 20.*u.deg

                    if cam_geom[tel_id].cam_id == "ASTRI":
                        pmt_signal = apply_mc_calibration_ASTRI(
                                        event.r0.tel[tel_id].adc_sums,
                                        event.mc.tel[tel_id].dc_to_pe,
                                        event.mc.tel[tel_id].pedestal)
                    else:
                        pmt_signal = apply_mc_calibration(
                            event.r0.tel[tel_id].adc_sums[0],
                            event.mc.tel[tel_id].dc_to_pe[0],
                            event.mc.tel[tel_id].pedestal[0])

                    max_signal = np.max(pmt_signal)

                    # trying to clean the image
                    try:
                        pmt_signal, new_geom = \
                            Cleaner.clean(pmt_signal.copy(), cam_geom[tel_id],
                                          event.inst.optical_foclen[tel_id])

                        if np.count_nonzero(pmt_signal) < 3:
                            continue

                    except FileNotFoundError as e:
                        print(e)
                        continue
                    except EdgeEventException:
                        continue

                    Imagecutflow.count("cleaning")

                    # trying to do the hillas reconstruction of the images
                    try:
                        moments = hillas_parameters(new_geom.pix_x,
                                                    new_geom.pix_y,
                                                    pmt_signal)

                        if not (moments.width > 0 and moments.length > 0):
                            continue

                    except HillasParameterizationError as e:
                        print(e)
                        print("ignoring this camera")
                        pass

                    hillas_dict[tel_id] = moments
                    tot_signal += moments.size

                    Imagecutflow.count("Hillas")

                if Eventcutflow.cut("min2Tels reco", len(hillas_dict)):
                    continue

                try:
                    # telescope loop done, now do the core fit
                    fit.get_great_circles(hillas_dict,
                                          event.inst,
                                          tel_phi, tel_theta)
                    pos_fit_cr, err_est_pos = fit.fit_core_crosses()
                except Exception as e:
                    print(e)
                    continue

                if Imagecutflow.cut("position nan", pos_fit_cr):
                    continue

                pos_fit = pos_fit_cr

                Eventcutflow.count("position fit")

                # now prepare the features for the classifier
                features_evt = {}
                NTels = len(hillas_dict)
                for tel_id in hillas_dict.keys():
                    Imagecutflow.count("pre-features")

                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                    moments = hillas_dict[tel_id]

                    impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                    impact_dist_rec = linalg.length(tel_pos-pos_fit)
                    features_tel = [
                                impact_dist_rec/u.m,
                                tot_signal,
                                max_signal,
                                moments.size,
                                NTels,
                                moments.cen_x/u.m,
                                moments.cen_y/u.m,
                                moments.width/u.m,
                                moments.length/u.m,
                                moments.skewness,
                                moments.kurtosis,
                                err_est_pos/u.m
                              ]

                    if Imagecutflow.cut("features nan", features_tel):
                        continue

                    cam_id = cam_geom[tel_id].cam_id
                    if cam_id in features_evt:
                        features_evt[cam_id] += [features_tel]
                    else:
                        features_evt[cam_id] = [features_tel]

                if len(features_evt):
                    Features_event_list.append(features_evt)
                    MC_Energies.append(mc_shower.energy)

                if signal_handler.stop:
                    break

    feature_labels = [
                        "impact_dist",
                        "sum_signal_evt",
                        "max_signal_cam",
                        "sum_signal_cam",
                        "NTels_rec",
                        "cen_x",
                        "cen_y",
                        "width",
                        "length",
                        "skewness",
                        "kurtosis",
                        "err_est_pos",
                      ]

    print()

    print("length of features:")
    print(len(Features_event_list))


    reg_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
                  'random_state': 0}

    # try neural network
    #from sklearn.neural_network import MLPClassifier
    #clf_kwargs = {'classifier': MLPClassifier, 'random_state': 1, 'alpha': 1e-5,
                  #'hidden_layer_sizes': (50,50,)}

    reg = fancy_EnergyRegressor(**reg_kwargs)
    reg.fit(Features_event_list, MC_Energies)

    print(reg)

    # save the regressor to disk
    if args.store:
        reg.save("{}/regressor_{}_{}_{}.pkl".format(args.outdir,
                        args.mode, args.raw.replace(" ", ""), reg))


    if args.plot:
        # extract and show the importance of the various training features
        try:
            reg.show_importances(feature_labels)
            plt.suptitle("{} ** {}".format(
                "wavelets" if args.mode == "wave" else "tailcuts",
                reg))
            if args.write:
                save_fig('{}/regression_importance_{}_{}_{}'.format(args.plots_dir,
                            args.mode, args.raw.replace(" ", ""), reg))
        except AttributeError as e:
            print("{} does not support feature importances".format(reg))


        # plot area under curve for a few cross-validations
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5)
        bins = np.array([np.linspace(-2, 3, 51), np.linspace(0, 4, 51)])
        myhist, _, _ = np.histogram2d([], [], bins=bins)
        X, y = np.array(Features_event_list), convert_astropy_array(MC_Energies)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            reg = fancy_EnergyRegressor(**reg_kwargs)

            reg.fit(X_train, y_train)

            y_score = reg.predict(X_test)

            myhist += np.histogram2d(np.log10(y_test/u.TeV), y_score/y_test, bins=bins)[0]

        plt.figure()
        plt.imshow(myhist, interpolation='none', origin='lower',
                   extent=bins[:, [0,-1]].ravel() )
        plt.xlabel('log10(E_MC / TeV)')
        plt.ylabel('E_predict / E_MC')
        #plt.gca().set_xscale("log")
        plt.suptitle("{} ** {}".format(
            "wavelets" if args.mode == "wave" else "tailcuts",
            reg))
        plt.legend(loc="lower right", title="different cross validations")

        if args.write:
            save_fig('{}/classification_area_under_curve_{}_{}_{}'.format(args.plots_dir,
                        args.mode, args.raw.replace(" ", ""), reg))


        plt.show()
