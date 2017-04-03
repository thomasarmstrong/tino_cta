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
from modules.EventClassifier import *


def get_class_string(cls):
    print(cls)
    return str(cls.__class__).split('.')[-1].split("'")[0]


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--outdir', type=str,
                        default='data/classify_pickle')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    args = parser.parse_args()

    filenamelist_gamma  = glob("{}/gamma/run*gz".format(args.indir))
    filenamelist_proton = glob("{}/proton/run*gz".format(args.indir))

    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # counting events and where they might have gone missing
    Eventcutflow = {"p": CutFlow("EventCutFlow Protons"),
                    "g": CutFlow("EventCutFlow Gammas")}
    Imagecutflow = {"p": CutFlow("ImageCutFlow Protons"),
                    "g": CutFlow("ImageCutFlow Gammas")}
    for E in Eventcutflow.values():
        E.set_cut("noCuts", None)
        E.set_cut("min2Tels trig", lambda x: x < 2)
        E.set_cut("min2Tels reco", lambda x: x < 2)

    # class that wraps tail cuts and wavelet cleaning
    Cleaner = ImageCleaner(mode=args.mode, wavelet_options=args.raw,
                           skip_edge_events=False)  # args.skip_edge_events)

    # simple hillas-based shower reco
    fit = FitGammaHillas()

    # wrapper for the scikit learn classifier
    classifier = EventClassifier(cutflow=Eventcutflow)

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    for filenamelist_class in [sorted(filenamelist_gamma)[:14],
                               sorted(filenamelist_proton)[:100]]:
        signal_handler.stop = False

        # get type of event for the classifier
        # assume that there are only gamma and proton as event class
        # if `filenamelist_gamma` is empty, though `cl` will be set to proton, the
        # `filename` loop will be empty, so no mislabelling will occur
        cl = "g" if "gamma" in " ".join(filenamelist_class) else "p"

        for filename in filenamelist_class[:args.last]:
            print("filename = {}".format(filename))

            source = hessio_event_source(filename,
                                         allowed_tels=allowed_tels,
                                         max_events=args.max_events)

            # event loop
            for event in source:

                Eventcutflow[cl].count("noCuts")

                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                if Eventcutflow[cl].cut("min2Tels trig", len(event.dl0.tels_with_data)):
                    continue

                # telescope loop
                tot_signal = 0
                max_signal = 0
                hillas_dict = {}
                for tel_id in event.dl0.tels_with_data:
                    Imagecutflow[cl].count("noCuts")

                    pmt_signal = apply_mc_calibration_ASTRI(
                                    event.r0.tel[tel_id].adc_sums,
                                    event.mc.tel[tel_id].dc_to_pe,
                                    event.mc.tel[tel_id].pedestal)

                    max_signal = np.max(pmt_signal)

                    # guessing camera geometry
                    if tel_id not in cam_geom:
                        cam_geom[tel_id] = CameraGeometry.guess(
                                            event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            event.inst.optical_foclen[tel_id])
                        tel_phi[tel_id] = 0.*u.deg
                        tel_theta[tel_id] = 20.*u.deg

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

                    Imagecutflow[cl].count("cleaning")

                    # trying to do the hillas reconstruction of the images
                    try:
                        moments = hillas_parameters(new_geom.pix_x,
                                                    new_geom.pix_y,
                                                    pmt_signal)

                        if not (moments.width > 0 and moments.length > 0):
                            continue

                        if False:
                            from ctapipe.visualization import CameraDisplay
                            fig = plt.figure(figsize=(17, 10))

                            ax1 = fig.add_subplot(121)
                            disp1 = CameraDisplay(cam_geom[tel_id],
                                                  image=np.sqrt(event.mc.tel[tel_id]
                                                                .photo_electron_image),
                                                  ax=ax1)
                            disp1.cmap = plt.cm.inferno
                            disp1.add_colorbar()
                            plt.title("sqrt photo-electron image")

                            ax3 = fig.add_subplot(122)
                            disp3 = CameraDisplay(new_geom,
                                                  image=np.sqrt(pmt_signal),
                                                  ax=ax3)
                            disp3.overlay_moments(moments, color='seagreen', linewidth=3)
                            disp3.cmap = plt.cm.inferno
                            disp3.add_colorbar()
                            plt.title("sqrt cleaned image")
                            plt.suptitle(args.mode)
                            plt.show()

                    except HillasParameterizationError as e:
                        print(e)
                        print("ignoring this camera")
                        pass

                    hillas_dict[tel_id] = moments
                    tot_signal += moments.size

                    Imagecutflow[cl].count("Hillas")

                if Eventcutflow[cl].cut("min2Tels reco", len(hillas_dict)):
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

                if np.isnan(pos_fit_cr.value).any():
                    continue

                pos_fit = pos_fit_cr

                Eventcutflow[cl].count("position fit")

                # now prepare the features for the classifier
                features_evt = {}
                NTels = len(hillas_dict)
                for tel_id in hillas_dict.keys():
                    Imagecutflow[cl].count("pre-features")

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
                                moments.width/u.m,
                                moments.length/u.m,
                                moments.skewness,
                                moments.kurtosis,
                                err_est_pos/u.m
                              ]
                    if np.isnan(features_tel).any():
                        continue

                    Imagecutflow[cl].count("features nan")

                    features_evt[tel_id] = features_tel
                if len(features_evt):
                    classifier.Features[cl].append(features_evt)
                    classifier.MCEnergy[cl].append(mc_shower.energy)

                if signal_handler.stop:
                    break
            if signal_handler.stop:
                break

    feature_labels = [
                        "impact_dist",
                        "sum_signal_evt",
                        "max_signal_cam",
                        "sum_signal_cam",
                        "NTels_rec",
                        "width",
                        "length",
                        "skewness",
                        "kurtosis",
                        "err_est_pos",
                      ]

    print()


    # reduce the number of events so that
    # they are the same in gammas and protons
    classifier.equalise_nevents()

    trainFeatures = []
    trainClasses  = []

    for cl in classifier.class_list:
        trainFeatures += classifier.Features[cl]
        trainClasses += [cl]*len(classifier.Features[cl])

    clf_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
                  'random_state': 0}

    # try neural network
    #from sklearn.neural_network import MLPClassifier
    #clf_kwargs = {'classifier': MLPClassifier, 'random_state': 1, 'alpha': 1e-5,
                  #'hidden_layer_sizes': (50,50,)}

    clf = fancy_EventClassifier(**clf_kwargs)
    clf.fit(trainFeatures, trainClasses)

    print(clf)

    # save the classifier to disk
    if args.store:
        clf.save("{}/classifier_{}_{}_{}.pkl".format(args.outdir,
                        args.mode, args.raw.replace(" ", ""), clf))


    if args.plot:
        # extract and show the importance of the various training features
        try:
            clf.show_importances(feature_labels)
            plt.suptitle("{} ** {}".format(
                "wavelets" if args.mode == "wave" else "tailcuts",
                clf))
            if args.write:
                save_fig('{}/classification_importance_{}_{}_{}'.format(args.plots_dir,
                            args.mode, args.raw.replace(" ", ""), clf))
        except AttributeError as e:
            print("{} classifier does not support feature importances".format(clf))

        # plot area under curve for a few cross-validations
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
        plt.figure()
        colours = ["darkorange", "r", "b", "g", "black"]

        X, y = np.array(trainFeatures), np.array(trainClasses)
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = fancy_EventClassifier(**clf_kwargs)

            clf.fit(X_train, y_train)

            y_score = clf.predict_proba(X_test)[:,0]

            fpr, tpr, _ = roc_curve(y_test == "g", y_score)
            roc_auc = auc(fpr, tpr)

            lw = 2
            plt.plot(fpr, tpr, color=colours[i],
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

        # plot a diagonal line that represents purely random choices
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.suptitle("{} ** {}".format(
            "wavelets" if args.mode == "wave" else "tailcuts",
            clf))
        plt.legend(loc="lower right", title="different cross validations")

        if args.write:
            save_fig('{}/classification_area_under_curve_{}_{}_{}'.format(args.plots_dir,
                        args.mode, args.raw.replace(" ", ""), clf))


        histos = {'g': None, 'p': None}
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.19, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = fancy_EventClassifier(**clf_kwargs)

            clf.fit(X_train, y_train)

            for cl in histos.keys():
                test = X_test[y_test==cl]
                NTels = [len(evt) for evt in test]

                try:
                    y_score = clf.predict_proba(test)[:,0]
                except:
                    continue

                temp = np.histogram2d(NTels, y_score, bins=(range(1, 10),
                                                            np.linspace(0, 1, 11)))[0].T

                if histos[cl] is None:
                    histos[cl] = temp
                else:
                    histos[cl] += temp

        fig = plt.figure()
        for cl in histos.keys():
            histo = histos[cl]

            ax = plt.subplot(121 if cl == "g" else 122)
            histo_normed = histo / histo.max(axis=0)
            im = ax.imshow(histo_normed, interpolation='none', origin='lower',
                            aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
            cb = fig.colorbar(im, ax=ax)
            ax.set_xlabel("NTels")
            ax.set_ylabel("drifted gammaness")
            plt.title(" ** ".join([args.mode, "protons" if cl=='p' else "gamma"]))

        plt.show()
