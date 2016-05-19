import numpy as np
from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from astropy import units as u
import sys


from numpy import ceil
from matplotlib import pyplot as plt
from ctapipe import visualization, io
def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("p               - Print all event data")
    print("i               - event Info")
    print("q               - Quit")
    return input("Choice: ")

fig = plt.figure(figsize=(12, 8))
def display_event(event, calibrate = 0, max_tel = 5):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    ntels = min(max_tel, len(event.dl0.tels_with_data))
    fig.clear()

    plt.suptitle("EVENT {}".format(event.dl0.event_id))

    disps = []

    for ii, tel_id in enumerate(event.dl0.tels_with_data):
        if ii >= max_tel: break
        print("\t draw cam {}...".format(tel_id))
        nn = int(ceil((ntels)**.5))
        ax = plt.subplot(nn, 2*nn, 2*(ii+1)-1)

        x, y = event.meta.pixel_pos[tel_id]
        geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
        disp = visualization.CameraDisplay(geom, ax=ax,
                                           title="CT{0} DetectorResponse".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = plt.cm.hot
        chan = 0
        signals = event.dl0.tel[tel_id].adc_sums[chan].astype(float)[:]
        if calibrate:
            signals = apply_mc_calibration(signals, tel_id)
        disp.image = signals
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)
        
        
        ax = plt.subplot(nn, 2*nn, 2*(ii+1))

        geom = io.CameraGeometry.guess(x, y, event.meta.optical_foclen[tel_id])
        disp = visualization.CameraDisplay(geom, ax=ax,
                                           title="CT{0} PhotoElectrons".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.cmap = plt.cm.hot
        chan = 0


        #print (event.mc.tel[tel_id].photo_electrons)
        for jj in range(len(event.mc.tel[tel_id].photo_electrons)):
            event.dl0.tel[tel_id].adc_sums[chan][jj] = event.mc.tel[tel_id].photo_electrons[jj]
        signals2 = event.dl0.tel[tel_id].adc_sums[chan].astype(float)
        disp.image = signals2
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disps.append(disp)
        

    return disps

import pyhessio
def get_mc_calibration_coeffs(tel_id):
    """
    Get the calibration coefficients from the MC data file to the
    data.  This is ahack (until we have a real data structure for the
    calibrated data), it should move into `ctapipe.io.hessio_event_source`.

    returns
    -------
    (peds,gains) : arrays of the pedestal and pe/dc ratios.
    """
    peds = pyhessio.get_pedestal(tel_id)[0]
    gains = pyhessio.get_calibration(tel_id)[0]
    return peds, gains


def apply_mc_calibration(adcs, tel_id):
    """
    apply basic calibration
    """
    peds, gains = get_mc_calibration_coeffs(tel_id)

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


if __name__ == '__main__':
    


    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-t','--tel', type=int)
    parser.add_argument('-m', '--max-events', type=int, default=10000)
    parser.add_argument('-w', '--write', action='store_true',
                        help='write images to files')
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-s', '--show-samples', action='store_true',
                        help='show time-variablity, one frame at a time')
    parser.add_argument('-c','--calibrate', action='store_true',
                        help='apply calibration coeffs from MC')
    args = parser.parse_args()

    if args.filename:
        filename = args.filename
    else:
        filenamelist = []
        filenamelist += glob("/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/gamma_20deg_180deg_run10*")
        filenamelist += glob("/home/ichanmich/software/cta/datafiles/*")
        filename = filenamelist[0]

    source = hessio_event_source(filename,
                                 #allowed_tels=[args.tel],
                                 #allowed_tels=[1,2,3,4,5],
                                 allowed_tels=[31,32,33],
                                 max_events=args.max_events)

    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        print(event.dl0.tels_with_data)
        if args.tel and args.tel not in event.dl0.tels_with_data: continue

                
        while True:
            response = get_input()
            print()
            if response.startswith("d"):
                disps = display_event(event)
                plt.pause(0.1)
            elif response.startswith("p"):
                print("--event-------------------")
                print(event)
                print("--event.dl0---------------")
                print(event.dl0)
                print("--event.dl0.tel-----------")
                for teldata in event.dl0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
            elif response.startswith('i'):
                for tel_id in sorted(event.dl0.tel):
                    for chan in event.dl0.tel[tel_id].adc_samples:
                        npix = len(event.meta.pixel_pos[tel_id][0])
                        print("CT{:4d} ch{} pixels:{} samples:{}"
                            .format(tel_id, chan, npix,
                                    event.dl0.tel[tel_id].
                                    adc_samples[chan].shape[1]))

            elif response.startswith('q'):
                sys.exit()
            else:
                sys.exit()

        if response.startswith('q'):
            sys.exit()
