#!/usr/bin/env python3

from helper_functions import *
from astropy import units as u
from collections import namedtuple

from sys import exit, path
from os.path import expandvars
from glob import glob
from itertools import chain

from ctapipe.utils import linalg

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from modules.CutFlow import *
from modules.ImageCleaning import *

try:
    from ctapipe.reco.event_classifier import *
    print("using ctapipe event_classifier")
except ImportError:
    from modules.event_classifier import *
    print("using tino_cta event_classifier")

try:
    from ctapipe.reco.FitGammaHillas import \
        FitGammaHillas as HillasReconstructor, TooFewTelescopesException
except ImportError:
    from ctapipe.reco.HillasReconstructor import \
        HillasReconstructor, TooFewTelescopesException

from modules.prepare_event import EventPreparator


pckl_write = True
pckl_load = not pckl_write

# for which cam_id to generate a models
cam_id_list = [
        # 'GATE',
        # 'HESSII',
        # 'NectarCam',
        # 'LSTCam',
        # 'SST-1m',
        'FlashCam',
        'ASTRICam',
        # 'SCTCam',
        ]

ClassifierFeatures = namedtuple("ClassifierFeatures", (
                                "impact_dist",
                                "sum_signal_evt",
                                "max_signal_cam",
                                "sum_signal_cam",
                                "N_LST",
                                "N_MST",
                                "N_SST",
                                "width",
                                "length",
                                "skewness",
                                "kurtosis",
                                "err_est_pos",
                                "err_est_dir"))


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--outpath', type=str,
                        default='data/classifier_pickle/classifier_prod3b'
                                '_{mode}_{cam_id}_{classifier}.pkl')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    args = parser.parse_args()

    filenamelist_gamma = sorted(glob("{}/gamma/*gz".format(args.indir)))
    filenamelist_proton = sorted(glob("{}/proton/*gz".format(args.indir)))

    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=False, island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    preper = EventPreparator(calib=None, cleaner=cleaner,
                             hillas_parameters=hillas_parameters, shower_reco=shower_reco,
                             event_cutflow=Eventcutflow, image_cutflow=Imagecutflow,
                             # event/image cuts:
                             allowed_cam_ids=[],  # [] or None means: all
                             min_ntel=2,
                             min_charge=args.min_charge, min_pixel=3)
    Imagecutflow.add_cut("features nan", lambda x: np.isnan(x).any())

    # features and targets for the training
    Features_event_list = {"g": [], "p": []}
    MC_Energies = {"g": [], "p": []}

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = None  # all telescopes
    # allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    # allowed_tels = np.arange(10).tolist() + np.arange(34, 41).tolist()
    allowed_tels = prod3b_tel_ids("F+A")
    for filenamelist_class in [filenamelist_gamma[:10],
                               filenamelist_proton[:60]]:

        if pckl_load:
            break

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

            # loop that cleans and parametrises the images and performs the reconstruction
            for (event, hillas_dict, n_tels,
                 tot_signal, max_signals, pos_fit, dir_fit,
                 err_est_pos, err_est_dir) in preper.prepare_event(source):

                # now prepare the features for the classifier
                features_evt = {}
                for tel_id in hillas_dict.keys():
                    Imagecutflow.count("pre-features")

                    moments = hillas_dict[tel_id]

                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m
                    impact_dist_rec = linalg.length(tel_pos-pos_fit)

                    features_tel = ClassifierFeatures(
                                impact_dist_rec/u.m,
                                tot_signal,
                                max_signals[tel_id],
                                moments.size,
                                n_tels["LST"],
                                n_tels["MST"],
                                n_tels["SST"],
                                moments.width/u.m,
                                moments.length/u.m,
                                moments.skewness,
                                moments.kurtosis,
                                err_est_pos/u.m,
                                err_est_dir/u.deg
                            )

                    if Imagecutflow.cut("features nan", features_tel):
                        continue

                    cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                    if cam_id in features_evt:
                        features_evt[cam_id] += [features_tel]
                    else:
                        features_evt[cam_id] = [features_tel]

                if len(features_evt):
                    Features_event_list[cl].append(features_evt)
                    MC_Energies[cl].append(event.mc.energy)

                if signal_handler.stop:
                    break
            if signal_handler.stop:
                break

    print()

    if pckl_load:
        print("reading pickle")
        from sklearn.externals import joblib
        Features_event_list = \
            joblib.load("./data/{}_classification_features_prod3b.pkl".format(args.mode))
    elif pckl_write:
        print("writing pickle")
        from sklearn.externals import joblib
        joblib.dump(Features_event_list,
                    "./data/{}_classification_features_prod3b.pkl".format(args.mode))

    print("number of events:")
    for cl, feat in Features_event_list.items():
        print(cl, len(feat))

    trainFeatures = []
    trainClasses = []

    # reduce the number of events by random draw so that they are the same
    # in gammas and protons; then add them to a single list
    minEvents = min([len(Features) for Features in Features_event_list.values()])
    for cl in Features_event_list.keys():
        # trainFeatures = np.append(
        #                     trainFeatures,
        #                     np.random.choice(Features_event_list[cl], minEvents, False))
        # trainClasses += [cl]*minEvents
        trainFeatures += Features_event_list[cl]
        trainClasses += [cl]*len(Features_event_list[cl])


    # use default random forest classifier
    clf_kwargs = {'n_estimators': 40, 'max_depth': None, 'min_samples_split': 2,
                  'random_state': 0}

    # try neural network
    if False:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        mlp_kwargs = {'random_state': 1, 'alpha': 1e-5,
                      'hidden_layer_sizes': (150, 150, 150)}
        mlp_clf = MLPClassifier(**mlp_kwargs)
        sskal = StandardScaler()

        clf_kwargs = {"classifier": Pipeline,
                      "steps": [("sskal", sskal), ("mlp_clf", mlp_clf)]}

    clf_kwargs['cam_id_list'] = cam_id_list

    clf = EventClassifier(**clf_kwargs)
    print(clf)

    train_features, train_classes = clf.reshuffle_event_list(trainFeatures, trainClasses)

    print("number of g:", np.count_nonzero(np.array(train_classes["ASTRICam"]) == 'g'))
    print("number of p:", np.count_nonzero(np.array(train_classes["ASTRICam"]) == 'p'))
    # exit()

    # save the classifier to disk
    if args.store:
        clf.fit(train_features, train_classes)
        clf.save(args.outpath.format(**{"mode": args.mode, "classifier": clf,
                                        "cam_id": "{cam_id}"}))

    if args.plot:
        # extract and show the importance of the various training features
        try:
            clf.fit(train_features, train_classes)
            clf.show_importances(ClassifierFeatures._fields)
            plt.suptitle("{} ** {}".format(
                "wavelets" if args.mode == "wave" else "tailcuts",
                clf))
            if args.write:
                save_fig('{}/classification_importance_{}_{}'.format(args.plots_dir,
                         args.mode, clf))
        except AttributeError as e:
            print(e)
            print("{} does not support feature importances".format(clf))

        # plot area under curve for a few cross-validations
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
        colours = ["darkorange", "r", "b", "g", "black"]

        plt.figure()
        lw = 2
        # plot a diagonal line that represents purely random choices
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.suptitle("{} ** {}".format(
            "wavelets" if args.mode == "wave" else "tailcuts", clf))

        X, y = np.array(trainFeatures), np.array(trainClasses)
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = EventClassifier(**clf_kwargs)

            X_train_flat, y_train_flat = clf.reshuffle_event_list(X_train, y_train)
            clf.fit(X_train_flat, y_train_flat)

            y_score = clf.predict_proba_by_event(X_test)[:, 0]

            fpr, tpr, _ = roc_curve(y_test == "g", y_score)
            roc_auc = auc(fpr, tpr)
            print("area under curve:", roc_auc)

            plt.plot(fpr, tpr, color=colours[i],
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.pause(.1)

        # legend does not update dynamically, needs to be called at the end
        plt.legend(loc="lower right", title="cross validation")

        if args.write:
            save_fig('{}/classification_receiver_operating_curve_{}_{}'
                     .format(args.plots_dir, args.mode, clf))
        plt.pause(.1)

        # plot gammaness as function of number of telescopes
        histos = {'g': None, 'p': None}
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.19, random_state=0)
        for i, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = EventClassifier(**clf_kwargs)

            clf.fit(*clf.reshuffle_event_list(X_train, y_train))

            for cl in histos.keys():
                test = X_test[y_test == cl]
                NTels = []
                for evt in test:
                    n_tels = 0
                    for c, t in evt.items():
                        n_tels += len(t)
                    NTels.append(n_tels)

                try:
                    y_score = clf.predict_proba_by_event(test)[:, 0]
                except:
                    continue

                temp = np.histogram2d(NTels, y_score,
                                      bins=(range(2, 10), np.linspace(0, 1, 11)))[0].T

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
            plt.title("protons" if cl == 'p' else "gamma")

        plt.suptitle("{} ** {}".format("wavelets" if args.mode == "wave" else "tailcuts",
                                       clf))
        plt.subplots_adjust(left=0.10, right=0.95, wspace=0.33)

        if args.write:
            save_fig('{}/classification_gammaness_vs_NTel_{}_{}'
                     .format(args.plots_dir, args.mode, clf))

        plt.show()
