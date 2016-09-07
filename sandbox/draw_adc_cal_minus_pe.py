from sys import exit
from glob import glob
import argparse

import numpy as np

from itertools import chain

import matplotlib.pyplot as plt

from astropy import units as u

from ctapipe.io.hessio import hessio_event_source


import signal
stop = None
def signal_handler(signal, frame):
    global stop
    if stop:
        print('you pressed Ctrl+C again -- exiting NOW')
        exit(-1)
    print('you pressed Ctrl+C!')
    print('exiting loop after this event')
    stop = True
signal.signal(signal.SIGINT, signal_handler)

import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id, mode=0, adc_tresh=3500):
    """
    apply basic calibration
    """
    peds0 = pyhessio.get_pedestal(tel_id)[0]
    peds1 = pyhessio.get_pedestal(tel_id)[1]
    gains0 = pyhessio.get_calibration(tel_id)[0]
    gains1 = pyhessio.get_calibration(tel_id)[1]
    
    
    if mode == 0:
        calibrated = [ (adc0- 971)*gain0 if adc0 < adc_tresh else (adc1- 961)*gain1 for adc0, adc1, gain0, gain1 in zip(adcs[0], adcs[1], gains0,gains1) ]
    else: 
        calibrated = [ (adc0-ped0)*gain0 if adc0 < adc_tresh else (adc1-ped1)*gain1 for adc0, adc1, gain0, gain1, ped0, ped1 in zip(adcs[0], adcs[1], gains0,gains1, peds0, peds1) ]
    
    return np.array(calibrated)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=None)
    parser.add_argument('-i', '--indir',    type=str, 
                        default="/local/home/tmichael/Data/cta/ASTRI9")
    parser.add_argument('-r', '--runnr',    type=str, default="*")
    parser.add_argument('-t', '--tel',      type=int, default=None)
    parser.add_argument('-o', '--outtoken', type=str, default=None)
    args = parser.parse_args()
    
    filenamelist_gamma  = glob( "{}/gamma/run*{}.simtel.gz".format(args.indir,args.runnr ))
    filenamelist_proton = [] #glob( "{}/proton/run{}.*gz".format(args.indir,args.runnr ))
    
    print(  "{}/gamma/run*{}.simtel.gz".format(args.indir,args.runnr ))
    if len(filenamelist_gamma) == 0:
        print("no gammas found")
        exit()
    
    allowed_tels = set(range(10))
    if args.tel: allowed_tels = allowed_tels & set([args.tel])

    adc_cal0  = []
    adc_cal1  = []
    residual0 = []
    residual1 = []
    
    
    for filename in chain(sorted(filenamelist_gamma)[:], sorted(filenamelist_proton)[:]):
        print("filename = {}".format(filename))
    
        source = hessio_event_source(filename,
                                allowed_tels=allowed_tels,
                                max_events=args.max_events)
    
        for event in source:
            tels = set(event.trig.tels_with_trigger) & set(event.dl0.tels_with_data)
            for tel_id in tels:
                dat0 = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id,0)
                dat1 = apply_mc_calibration_ASTRI(event.dl0.tel[tel_id].adc_sums, tel_id,1)
                
                MoCa = event.mc.tel[tel_id].photo_electrons
                
                adc_cal0 .append( dat0[0] )
                adc_cal1 .append( dat1[0] )
                residual0.append( adc_cal0[-1] - MoCa[0] )
                residual1.append( adc_cal1[-1] - MoCa[0] )
            if stop: break
        if stop: break
    
    fig = plt.figure()
    
    bins = np.linspace(-5, 10, 75)
    
    plt.hist(residual0,bins, alpha=0.5, label='corrected calibration')
    plt.hist(residual1,bins, alpha=0.5, label='mixed calibration')
    plt.legend(loc='upper right')
    plt.show()