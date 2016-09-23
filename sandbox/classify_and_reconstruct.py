from sys import exit, path


path.append("/local/home/tmichael/software/jeremie_cta/snippets/ctapipe")

path.append("/local/home/tmichael/software/jeremie_cta/data-pipeline-standalone-scripts")
from datapipe.classifiers.EventClassifier import EventClassifier, apply_mc_calibration_ASTRI
from datapipe.reco.FitGammaHillas import FitGammaHillas


from glob import glob
import argparse

import matplotlib.pyplot as plt

from itertools import chain

import numpy as np

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils import linalg 
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
    parser.add_argument('-p', '--store_path', type=str, default='classifier/classifier.pkl')
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
    agree_threshold=.75
    min_tel = 3
    
    instrument_description = load_hessio(filenamelist_gamma[0])


    classifier = EventClassifier()
    classifier.setup_geometry(*instrument_description)
    #classifier.load(args.store_path+"_"+args.mode+".pkl")
    classifier.load(args.store_path)


    fit = FitGammaHillas()
    fit.setup_geometry( *instrument_description )

    
    source_orig = None
    fit_origs   = {'g':[], 'p':[]}
    
    for filenamelist_class in [ filenamelist_gamma, filenamelist_proton ]:
    #for filenamelist_class in [ filenamelist_proton ]:
        
        if "gamma" in filenamelist_class[0]: cl = "g"
        else:                                cl = "p"

        
        for filename in sorted(filenamelist_class)[:args.last]:
            print("filename = {}".format(filename))
            
            source = hessio_event_source(filename,
                                        allowed_tels=range(10), # smallest ASTRI aray
                                        #allowed_tels=range(34), # all ASTRI telescopes
                                        max_events=args.max_events)

            

            
            for event in source:
                    
                if source_orig is None:
                    shower = event.mc
                    # corsika measures azimuth the other way around, using phi=-az
                    source_dir = linalg.set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
                    # shower direction is downwards, shower origin up
                    source_orig = -source_dir
                    
                features = classifier.get_features(event, mode=args.mode, skip_edge_events=True)
                if len(features) < 3: continue
                
                predict  = classifier.predict(features)
                
                isGamma = [ 1 if (tel == "g") else 0 for tel in predict ]
                if np.mean(isGamma) > agree_threshold and len(isGamma) >= min_tel:

                    tel_data = {}
                    #for tel_id in set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data):
                        #data = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id)
                        #tel_data[tel_id] = data
                    for tel_id, tel in event.mc.tel.items():
                        tel_data[tel_id] = tel.photo_electrons
                        
                        
                    fit.get_great_circles(tel_data)
                    result1, crossings = fit.fit_origin_crosses()
                    result2            = fit.fit_origin_minimise(result1)
                    
                    fit_origs[cl].append(result2)
                

                
                if stop: break
            if stop:
                stop = False
                break
            
    
    off_angles      = {'p':[], 'g':[]}
    off_angles_sine = {'p':[], 'g':[]}
    off_angles_sq   = {'p':[], 'g':[]}
    
    for cl, cl_fits in fit_origs.items():
        for fit_orig in cl_fits:
            off_angles     [cl].append(        linalg.angle(fit_orig, source_orig).value )
            off_angles_sq  [cl].append(       (linalg.angle(fit_orig, source_orig).value**2 ) )
            off_angles_sine[cl].append( np.sin(linalg.angle(fit_orig, source_orig) ) )
    
    
    # TODO determine weights better (NEvents generated, signal/background flux ratios...)
    weight_g = 1
    weight_p = 1e5

    fig = plt.figure()
    plt.subplot(211)
    plt.hist([off_angles['p'],off_angles['g']], weights=[[weight_g]*len(off_angles['g']), [weight_p]*len(off_angles['p'])], rwidth=1, bins=10,stacked=True)#range=(0,.5))
    plt.xlabel("alpha")
    
    plt.subplot(212)
    plt.hist([off_angles_sq['p'],off_angles_sq['g']], rwidth=1, bins=10,stacked=True)#range=(0,.01))
    plt.xlabel("sin(alpha)")
    
    plt.show()