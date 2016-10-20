from sys import exit, path

path.append("/local/home/tmichael/software/jeremie_cta/snippets/ctapipe")

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")
from datapipe.classifiers.EventClassifier import EventClassifier


from glob import glob
import argparse

from itertools import chain

from ctapipe.io.hessio import hessio_event_source

from ctapipe.instrument.InstrumentDescription import load_hessio


import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting current loop after this event')
    stop = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/Data/cta/ASTRI9")
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('-t', '--teltype', type=str, default="SST_ASTRI")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave")
    parser.add_argument('-d', '--dry', dest='last', action='store_const',
                        const=1, default=-1),
    parser.add_argument('-s', '--store', action='store_true')
    parser.add_argument('-p', '--store_path', type=str, default='classifier/classifier')
    parser.add_argument('-c', '--self_check', action='store_true')
    args = parser.parse_args()
    
    filenamelist_gamma  = glob( "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    filenamelist_proton = glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))
    
    print(  "{}/gamma/run{}.*gz".format(args.indir,args.runnr ))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    if len(filenamelist_proton) == 0:
        print("no protons found")
        exit()

    signal.signal(signal.SIGINT, signal_handler)


    classifier = EventClassifier()
    classifier.setup_geometry(*load_hessio(filenamelist_gamma[0]))
    
    
    events = {'g':0, 'p':0}
    
    for filenamelist_class in [ filenamelist_gamma, filenamelist_proton ]:
        for filename in sorted(filenamelist_class)[:args.last]:
            print("filename = {}".format(filename))
            
            source = hessio_event_source(filename,
                                        allowed_tels=range(10), # smallest ASTRI aray
                                        #allowed_tels=range(34), # all ASTRI telescopes
                                        max_events=args.max_events)

            
            if filename in filenamelist_proton:
                Class = "p"
            else:
                Class = "g"

            
            for event in source:
                classifier.get_event(event, Class, skip_edge_events=True,mode=args.mode)
                events[Class] += 1
                
                if stop: break
            if stop:
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
    
    
    # reduce the number of events so that they are the same in gammas and protons
    NEvents = min(lengths.values())
    classifier.equalise_nevents(NEvents)



    
    if args.store:
        classifier.learn()
        classifier.save(args.store_path+"_"+args.mode+".pkl")
    
    if args.self_check:
        classifier.self_check(min_tel=4)



    
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


