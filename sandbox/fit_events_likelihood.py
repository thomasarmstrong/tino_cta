from sys import exit
from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from FitGammaLikelihood import FitGammaLikelihood
from astropy import units as u

from ctapipe.io.containers import MCShowerData as MCShower

from ctapipe.instrument.InstrumentDescription import load_hessio

from Telescope_Mask import TelDict



import signal
stop = None
stop = True
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting after current event')
    stop = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/")
    parser.add_argument('-r', '--runnr',   type=str, default="2?")
    parser.add_argument('-t', '--teltype', type=str, default="LST")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist = glob( "{}*run{}*gz".format(args.indir,args.runnr ))
    if len(filenamelist) == 0: 
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
        
    fit = FitGammaLikelihood()
    fit.set_atmosphere("atmprof_paranal.dat")
    fit.set_instrument_description( *load_hessio(filenamelist[0]) )
    fit.read_raw('pdf/LST_47_raw.npz')
    fit.normalise()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    for filename in sorted(filenamelist):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    # for now use only identical telescopes...
                                    allowed_tels=TelDict[args.teltype],
                                    max_events=args.max_events)

        for event in source:
            
            if len(event.dl0.tels_with_data) < 2: continue
            
            print('Scanning input file... count = {}'.format(event.count))
            print('available telscopes: {}'.format(event.dl0.tels_with_data))


            

            seed = MCShower()
            seed.energy       = event.mc.energy
            seed.alt          = event.mc.alt  + .1 * u.rad
            seed.az           = event.mc.az  # + 10 * u.deg
            seed.core_x       = event.mc.core_x
            seed.core_y       = event.mc.core_y
            
            # if not done yet, add the height of the shower maximum as a field to the shower container
            try:
                seed.add_item("h_shower_max")
            except AttributeError:
                pass
            # find the height of the shower maximum and set it in the shower container
            seed.h_shower_max = fit.shower_max_estimator.find_shower_max_height(event.mc.energy, event.mc.h_first_int, event.mc.alt)
            
            fit.seed = seed
            fit_result = fit.fit(event=event).x
            
            reco_shower = MCShower()
            reco_shower.energy       = fit_result[0] * u.TeV
            reco_shower.alt          = fit_result[1] * u.rad
            reco_shower.az           = fit_result[2] * u.rad
            reco_shower.core_x       = fit_result[3] * u.m
            reco_shower.core_y       = fit_result[4] * u.m
            reco_shower.h_shower_max = fit_result[5] * u.m
            
            print("seed shower:", seed)
            print("reco_shower:", reco_shower)
            print("MC shower:  ", event.mc)
                      
            if stop: break
        if stop: break  