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

import pyhessio

import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting current class loop after this file')
    stop = True


def convert_astropy_array(arr,unit=None):
    if unit is None: unit = arr[0].unit
    return [a / unit for a in arr]*unit
    

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
    parser.add_argument('-w', '--write', action='store_true')
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
    min_tel = 2
    
    instrument_description = load_hessio(filenamelist_gamma[0])


    classifier = EventClassifier()
    classifier.setup_geometry(*instrument_description)
    #classifier.load(args.store_path+"_"+args.mode+".pkl")
    classifier.load(args.store_path)


    fit = FitGammaHillas()
    fit.setup_geometry( *instrument_description )

    
    source_orig = None
    fit_origs   = {'g':[], 'p':[]}
    MC_Energy   = {'g':[], 'p':[]}
    
    events_total         = {'g':0, 'p':0}
    events_passd_telcut1 = {'g':0, 'p':0}
    events_passd_telcut2 = {'g':0, 'p':0}
    events_passd_gsel    = {'g':0, 'p':0}
    telescopes_total     = {'g':0, 'p':0}
    telescopes_passd     = {'g':0, 'p':0}
    
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
                events_total[cl] += 1
                    
                if source_orig is None:
                    shower = event.mc
                    ''' corsika measures azimuth the other way around, using phi=-az '''
                    source_dir = linalg.set_phi_theta(-shower.az, 90.*u.deg+shower.alt)
                    ''' shower direction is downwards, shower origin up '''
                    source_orig = -source_dir
                
                ''' skip events with less than minimum hit telescopes '''
                if len(set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data)) < min_tel: continue
                events_passd_telcut1[cl] += 1
                
                features = classifier.get_features(event, mode=args.mode, skip_edge_events=True)
                
                telescopes_total[cl] += len(set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data))
                telescopes_passd[cl] += len(features)
                
                try:
                    predict  = classifier.predict(features)
                except Exception as e:
                    print("error: ", e)
                    print("Ntels: {}, Nfeatures: {}".format(len(set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data)), len(features)))
                    print("skipping event")
                    continue
                
                isGamma = [ 1 if (tel == "g") else 0 for tel in predict ]

                ''' skip events with less than minimum hit telescopes where the event is not on the edge '''
                if len(isGamma) < min_tel: continue
                events_passd_telcut2[cl] += 1
                ''' skip events where too few classifiers agree it's a gamma '''
                if np.mean(isGamma) <= agree_threshold: continue
                events_passd_gsel[cl] += 1

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
                MC_Energy[cl].append(event.mc.energy.to(u.GeV).value)
                
            if stop:
                stop = False
                break
            
    
    off_angles = {'p':[], 'g':[]}
    for cl, cl_fits in fit_origs.items():
        for fit_orig in cl_fits:
            off_angles[cl].append( linalg.angle(fit_orig, source_orig).value )
        off_angles[cl] = np.array(off_angles[cl])
        
    for cl in ['g', 'p']:
        print(cl)
        print("telescopes_total: {}, telescopes_passd: {}, passed/total: {}".format(telescopes_total[cl], telescopes_passd[cl], telescopes_passd[cl]/telescopes_total[cl]))
        print("events_total: {},\n"
                "events_passd_telcut1: {}, passed/total telcut1: {},\n"
                "events_passd_telcut2: {}, passed/total telcut2: {},\n"
                "events_passd_gsel: {}, passed/total gsel: {} \n"
                "passd gsel / passd telcut: {}"
                .format(events_total[cl], 
                        events_passd_telcut1[cl], events_passd_telcut1[cl]/events_total[cl] if events_total[cl] > 0 else 0,
                        events_passd_telcut2[cl], events_passd_telcut2[cl]/events_total[cl] if events_total[cl] > 0 else 0,
                        events_passd_gsel[cl],    events_passd_gsel   [cl]/events_total[cl] if events_total[cl] > 0 else 0,
                        events_passd_gsel[cl]/events_passd_telcut2[cl] if events_passd_telcut2[cl] > 0 else 0))
        print()
    
    print("selected {} gammas and {} proton events".format( len(fit_origs['g']), len(fit_origs['p']) ) )
    
    weight_g = 1
    weight_p = 1e5

    
    phi = {'g':[], 'p':[] }
    the = {'g':[], 'p':[] }
    for cl in ['g']:
        for fit in fit_origs[cl]:
            phithe = linalg.get_phi_theta(fit)
            phi[cl].append(phithe[0] if phithe[0] > 0 else phithe[0]+360*u.deg)
            the[cl].append(phithe[1])
        
    if args.write:
        from astropy.table import Table
        for cl in ['g', 'p']:
            Table([ off_angles[cl], MC_Energy[cl]*u.GeV, phi[cl], the[cl] ], 
                  names=("off_angles", "MC_Energy", "phi", "theta") ).write("selected_events_"+cl+".fits",overwrite=True)


    fig = plt.figure()
    plt.subplot(311)
    plt.hist([off_angles['p'],off_angles['g']], weights=[[weight_p]*len(off_angles['p']), [weight_g]*len(off_angles['g'])], rwidth=1, bins=50,stacked=True)
    plt.xlabel("alpha")
    
    plt.subplot(312)
    plt.hist([ off_angles['p']**2, off_angles['g']**2 ], weights=[[weight_p]*len(off_angles['p']), [weight_g]*len(off_angles['g'])], rwidth=1, bins=50,stacked=True)
    plt.xlabel("alpha²")
    
    plt.subplot(313)
    plt.hist([ -np.cos(off_angles['p']), -np.cos(off_angles['g']) ], weights=[[weight_p]*len(off_angles['p']), [weight_g]*len(off_angles['g'])], rwidth=1, bins=50,stacked=True)
    plt.xlabel("-cos(alpha)")
    
    plt.pause(.1)
    
    
    
    fig2 = plt.figure()
    unit = u.deg
    plt.hist2d( convert_astropy_array(chain(phi['p'],phi['g']),unit),  convert_astropy_array(chain(the['p'],the['g']),unit),range=([ [(180-3), (180+3)], [(20-3),(20+3)] ]*u.deg).to(unit).value  )
    plt.xlabel("phi / {}".format(unit))
    plt.ylabel("theta / {}".format(unit))
    plt.show()