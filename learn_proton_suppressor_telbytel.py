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



if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--outdir', type=str,
                        default='data/classify_pickle')
    parser.add_argument('--figdir', type=str,
                        default='plots')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    parser.add_argument('--store', action='store_true',
                        help="save the classifier as pickled data")
    parser.add_argument('--sig', type=str, default="3",
                        help="significance parameter for wavelet cleaning")
    args = parser.parse_args()

    filenamelist_gamma  = glob("{}/gamma/run{}.*gz"
                               .format(args.indir, args.runnr))
    filenamelist_proton = glob("{}/proton/run{}.*gz"
                               .format(args.indir, args.runnr))

    print("{}/gamma/run{}.*gz".format(args.indir, args.runnr))
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

    # wrapper for the scikit learn classifier
    classifier = EventClassifier()

    # simple hillas-based shower reco
    fit = FitGammaHillas()

    # counting events and where they might have gone missing
    Eventcutflow = {"p": CutFlow("EventCutFlow"),
                    "g": CutFlow("EventCutFlow")}
    Imagecutflow = {"p": CutFlow("ImageCutFlow"),
                    "g": CutFlow("ImageCutFlow")}

    for E in Eventcutflow.values():
        E.set_cut("noCuts", None)
        E.set_cut("min2Tels", lambda x: x >= 2)

    classifier.Eventcutflow = Eventcutflow

    # class that wraps tail cuts and wavelet cleaning
    Cleaner_reco = ImageCleaner(mode=args.mode)
    Cleaner_clss = ImageCleaner(mode=args.mode,
                                wavelet_options="-K -C1 -m3 -s{} -n4".format(args.sig))

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    for filenamelist_class in [sorted(filenamelist_gamma)[:1],
                               sorted(filenamelist_proton)[:10]]:
        signal_handler.stop = False
        for filename in filenamelist_class[:args.last]:
            print("filename = {}".format(filename))

            source = hessio_event_source(filename,
                                         allowed_tels=allowed_tels,
                                         max_events=args.max_events)

            '''
            get type of event for the classifier '''
            cl = "g" if "gamma" in filenamelist_class[0] else "p"

            '''
            event loop '''
            for event in source:

                Eventcutflow[cl].count("noCuts")

                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                '''
                telescope loop '''
                tot_signal = 0
                max_signal = 0
                hillas_dict_reco = {}
                hillas_dict_clss = {}
                for tel_id in event.dl0.tels_with_data:
                    Imagecutflow[cl].count("noCuts")

                    pmt_signal = apply_mc_calibration_ASTRI(
                                    event.dl0.tel[tel_id].adc_sums,
                                    event.mc.tel[tel_id].dc_to_pe,
                                    event.mc.tel[tel_id].pedestal)

                    max_signal = np.max(pmt_signal)

                    '''
                    guessing camera geometry '''
                    if tel_id not in cam_geom:
                        cam_geom[tel_id] = CameraGeometry.guess(
                                            event.inst.pixel_pos[tel_id][0],
                                            event.inst.pixel_pos[tel_id][1],
                                            event.inst.optical_foclen[tel_id])
                        tel_phi[tel_id] = 0.*u.deg
                        tel_theta[tel_id] = 20.*u.deg

                    '''
                    trying to clean the image '''
                    try:
                        pmt_signal_reco, new_geom = \
                            Cleaner_reco.clean(pmt_signal.copy(), cam_geom[tel_id],
                                               event.inst.optical_foclen[tel_id])
                        pmt_signal_clss = \
                            Cleaner_clss.clean(pmt_signal, cam_geom[tel_id],
                                               event.inst.optical_foclen[tel_id])[0]
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    except EdgeEventException:
                        continue

                    Imagecutflow[cl].count("cleaning")

                    # trying to do the hillas reconstruction of the images
                    try:
                        moments_reco = hillas_parameters(new_geom.pix_x,
                                                         new_geom.pix_y,
                                                         pmt_signal_reco)
                        moments_clss = hillas_parameters(new_geom.pix_x,
                                                         new_geom.pix_y,
                                                         pmt_signal_clss)

                        if moments_reco.width < 1e-5 * u.m or \
                           moments_reco.length < 1e-5 * u.m:
                            continue
                        if np.isnan([moments_reco.width.value,
                                     moments_reco.length.value]).any():
                            continue

                        hillas_dict_reco[tel_id] = moments_reco
                        hillas_dict_clss[tel_id] = moments_clss
                        tot_signal += moments_clss.size

                    except HillasParameterizationError as e:
                        print(e)
                        print("ignoring this camera")
                        pass

                    Imagecutflow[cl].count("Hillas")

                if not Eventcutflow[cl].cut("min2Tels", len(hillas_dict_reco)):
                    continue

                try:
                    # telescope loop done, now do the core fit
                    fit.get_great_circles(hillas_dict_reco,
                                          event.inst,
                                          tel_phi, tel_theta)
                    pos_fit_cr = fit.fit_core_crosses()
                except Exception as e:
                    print(e)
                    continue

                if np.isnan(pos_fit_cr).any():
                    continue

                pos_fit = pos_fit_cr

                Eventcutflow[cl].count("position fit")

                # now prepare the features for the classifier
                features = []
                NTels = len(hillas_dict_clss)
                for tel_id in hillas_dict_clss.keys():
                    Imagecutflow[cl].count("pre-features")

                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                    moments = hillas_dict_clss[tel_id]

                    impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                    impact_dist_rec = linalg.length(tel_pos-pos_fit)
                    feature = [
                                # impact_dist_sim / u.m,
                                impact_dist_rec / u.m,
                                impact_dist_rec / u.m,
                                tot_signal,
                                max_signal,
                                moments.size,
                                NTels,
                                moments.width/u.m,
                                moments.length/u.m,
                                moments.skewness,
                                moments.kurtosis
                              ]
                    if np.isnan(feature).any():
                        continue

                    Imagecutflow[cl].count("features nan")

                    features.append(feature)
                if len(features):
                    classifier.Features[cl].append(features)
                    classifier.MCEnergy[cl].append(mc_shower.energy)

                if signal_handler.stop:
                    break
            if signal_handler.stop:
                break

    feature_labels = ["impact_dist",
                      "tot_signal",
                      "max_signal",
                      "size",
                      "NTels",
                      "width",
                      "length",
                      "skewness",
                      "kurtosis"
                      ]

    print()

    lengths = {}
    for cl in classifier.class_list:
        lengths[cl] = len(classifier.Features[cl])

    '''
    reduce the number of events so that
    they are the same in gammas and protons '''
    NEvents = min(lengths.values())
    classifier.equalise_nevents(NEvents)

    '''
    extract and show the importance of the various training features '''
    try:
        classifier.show_importances(feature_labels)
    except ValueError:
        for k in classifier.class_list:
            for a in classifier.Features[k]:
                print(a)

    if args.write:
        save_fig('{}/classification_importance_{}_rec-rec_c{}'
                 .format(args.figdir, args.mode, args.min_charge))
    plt.pause(.5)

    if args.store:
        classifier.learn()
        classifier.save("{}/classifier_{}_rec-rec-dist_c{}.pkl"
                        .format(args.outdir, args.mode, args.min_charge))


    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
                        #solver='lbfgs',  # default: 'adam'
                        random_state=1,
                        alpha=1e-5,
                        #hidden_layer_sizes=(5, 2)  # default: (100,)
                        )
    # clf = None

    if args.check:
        classifier.self_check(min_tel=4, split_size=10, clf=clf)
        plt.tight_layout()

        if args.write:
            save_fig('{}/classification_performance_{}_rec-rec_c{}'
                     .format(args.figdir, args.mode, args.min_charge))

        plt.pause(.1)

    for cl in classifier.class_list:
        print()
        print()
        print("Gamma" if cl == "g" else "Proton")
        try:
            Eventcutflow[cl]("to classify")
        except UndefinedCutException as e:
            print(e)
            Eventcutflow[cl]()
        print()
        Imagecutflow[cl]()

    plt.show()

    ##from sklearn.model_selection import train_test_split
    ##from sklearn.preprocessing import StandardScaler
    ##from sklearn.datasets import make_moons, make_circles, make_classification

    ##from sklearn.neural_network import MLPClassifier
    #from sklearn.neighbors import KNeighborsClassifier
    #from sklearn.svm import SVC
    ##from sklearn.gaussian_process import GaussianProcessClassifier
    ##from sklearn.gaussian_process.kernels import RBF
    #from sklearn.tree import DecisionTreeClassifier
    #from sklearn.ensemble import AdaBoostClassifier
    #from sklearn.naive_bayes import GaussianNB
    #from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    #from sklearn.ensemble import RandomForestClassifier
    #from sklearn.ensemble import ExtraTreesClassifier
    #from sklearn import svm

    #for clf in [KNeighborsClassifier(3),
                #SVC(kernel="linear", C=0.025),
                #SVC(gamma=2, C=1),
                ##GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                #DecisionTreeClassifier(max_depth=5),
                #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                ##MLPClassifier(alpha=1),
                #AdaBoostClassifier(),
                #GaussianNB(),
                #QuadraticDiscriminantAnalysis()]:
        #classifier.learn(clf)

        #for cl in classifier.Features.keys():
            #trainFeatures   = []
            #trainClasses    = []
            #for ev in classifier.Features[cl]:
                #trainFeatures += ev
                #trainClasses  += [cl]*len(ev)

            #print(cl,"score:", classifier.clf.score(trainFeatures, trainClasses) )
        #print()


