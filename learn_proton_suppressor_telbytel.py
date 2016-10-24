from helper_functions import *

from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline/"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe/"))
from datapipe.classifiers.EventClassifier import EventClassifier


from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException


from glob import glob
import argparse

from itertools import chain

from ctapipe.utils import linalg
from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio

from modules.ImageCleaning import ImageCleaner, \
                                  EdgeEventException, UnknownModeException
from modules.CutFlow import CutFlow

from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--out_path', type=str,
                        default='data/classify_pickle/classifier')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--store', action='store_true')
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

    '''
    prepare InstrumentDescription '''
    InstrDesc = load_hessio(filenamelist_gamma[0])

    '''
    wrapper for the scikit learn classifier '''
    classifier = EventClassifier()
    classifier.setup_geometry(*InstrDesc,
                              phi=180*u.deg, theta=20*u.deg)
    classifier.cleaner = ImageCleaner(args.mode)

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

    events = {'g':0, 'p':0}

    for filenamelist_class in [filenamelist_gamma, filenamelist_proton]:
        for filename in sorted(filenamelist_class)[:args.last]:
            print("filename = {}".format(filename))

            source = hessio_event_source(
                        filename,
                        allowed_tels=range(10),  # smallest ASTRI aray
                        # allowed_tels=range(34),  # all ASTRI telescopes
                        max_events=args.max_events)

            '''
            get type of event for the classifier '''
            if filename in filenamelist_proton:
                Class = "p"
            else:
                Class = "g"

            '''
            event loop '''
            for event in source:
                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                '''
                telescope loop '''
                tot_signal = 0
                hillas_dict1 = {}
                hillas_dict2 = {}
                for tel_id in set(event.trig.tels_with_trigger) & \
                              set(event.dl0.tels_with_data):
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
                if len(features):
                    classifier.Features[Class].append(features)
                    classifier.MCEnergy[Class].append(mc_shower.energy)
                    classifier.total_images += len(features)
                    events[Class] += 1

                if signal_handler.stop:
                    break
            if signal_handler.stop:
                stop = False
                break


    print("total images:", classifier.total_images)
    print("selected images:", classifier.selected_images)
    print()

    lengths = {}
    print("events:")
    for cl in classifier.class_list:
        lengths[cl] = len(classifier.Features[cl])
        print("found {}: {}".format(cl, events[cl]))
        print("pickd {}: {}".format(cl, len(classifier.Features[cl])))

    '''
    reduce the number of events so that
    they are the same in gammas and protons '''
    NEvents = min(lengths.values())
    classifier.equalise_nevents(NEvents)

    '''
    extract and show the importance of the various training features '''
    classifier.show_importances()
    plt.pause(.5)

    if args.store:
        classifier.learn()
        classifier.save(args.out_path+"_"+args.mode+"_rec-sim-dist.pkl")

    if args.check:
        classifier.self_check(min_tel=4, split_size=50, write=args.write,
                              out_token=args.mode+"_rec-sim-dist")




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


