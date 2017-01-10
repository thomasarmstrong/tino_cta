from helper_functions import *

from sys import exit, path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline/"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe/"))


from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException


from glob import glob
import argparse

from itertools import chain

from ctapipe.utils import linalg

from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_1 as hillas_parameters

from modules.ImageCleaning import ImageCleaner, \
                                  EdgeEventException, UnknownModeException
from modules.CutFlow import CutFlow
from modules.EventClassifier import EventClassifier


if __name__ == '__main__':

    parser = make_argparser()
    parser.add_argument('-o', '--out_dir', type=str,
                        default='data/classify_pickle/classifier')
    parser.add_argument('--check', action='store_true',
                        help="run a self check on the classification")
    parser.add_argument('--store', action='store_true',
                        help="save the classifier as pickled data")
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

    '''
    wrapper for the scikit learn classifier '''
    classifier = EventClassifier()

    '''
    simple hillas-based shower reco '''
    fit = FitGammaHillas()

    '''
    counting events and where they might have gone missing '''
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    '''
    class that wraps tail cuts and wavelet cleaning '''
    Cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow)

    '''
    catch ctr-c signal to exit current loop and still display results '''
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    events = {'g': 0, 'p': 0}

    allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    for filenamelist_class in [sorted(filenamelist_gamma)[:10],
                               sorted(filenamelist_proton)[:50]]:
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
                mc_shower = event.mc
                mc_shower_core = np.array([mc_shower.core_x.value,
                                           mc_shower.core_y.value]) * u.m

                '''
                telescope loop '''
                tot_signal = 0
                max_signal = 0
                hillas_dict = {}
                for tel_id in event.dl0.tels_with_data:
                    classifier.total_images += 1

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
                        tel_phi[tel_id] = 180.*u.deg
                        tel_theta[tel_id] = 20.*u.deg



                    '''
                    trying to clean the image '''
                    try:
                        pmt_signal, new_geom = \
                            Cleaner.clean(pmt_signal, cam_geom[tel_id],
                                          event.inst.optical_foclen[tel_id])
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    except EdgeEventException:
                        continue

                    '''
                    trying to do the hillas reconstruction of the images '''
                    try:
                        moments = hillas_parameters(new_geom.pix_x,
                                                    new_geom.pix_y,
                                                    pmt_signal)

                        hillas_dict[tel_id] = moments
                        tot_signal += moments.size

                    except HillasParameterizationError as e:
                        print(e)
                        print("ignoring this camera")
                        pass

                '''
                telescope loop done, now do the core fit '''
                fit.get_great_circles(hillas_dict,
                                      event.inst, tel_phi, tel_theta)
                seed = [0, 0]*u.m
                pos_fit = fit.fit_core(seed)

                '''
                now prepare the features for the classifier '''
                features = []
                NTels = len(hillas_dict)
                for tel_id in hillas_dict.keys():
                    tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                    moments = hillas_dict[tel_id]

                    impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                    impact_dist_rec = linalg.length(tel_pos-pos_fit)
                    features.append([
                                impact_dist_rec / u.m,
                                impact_dist_sim / u.m,
                                tot_signal,
                                max_signal,
                                moments.size,
                                NTels,
                                moments.width, moments.length,
                                moments.skewness,
                                moments.kurtosis,
                                moments.asymmetry
                                ])
                if len(features):
                    classifier.Features[cl].append(features)
                    classifier.MCEnergy[cl].append(mc_shower.energy)
                    classifier.selected_images += len(features)
                    events[cl] += 1

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
                      "kurtosis",
                      "asymmetry"
                      ]

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
    classifier.show_importances(feature_labels)
    plt.pause(.5)

    if args.store:
        classifier.learn()
        classifier.save(args.out_dir+"_"+args.mode+"_rec-sim-dist.pkl")

    if args.check:
        classifier.self_check(min_tel=4, split_size=10, write=args.write,
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


