from sys import exit
from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from FitGammaLikelihood import FitGammaLikelihood


#from astropy import units as u
#from ctapipe.io import CameraGeometry
#from guessPixDirection import *
#tel_phi   =   0.*u.rad
#tel_theta =  20.*u.deg


from ctapipe.instrument.InstrumentDescription import load_hessio

from Telescope_Mask import TelDict



import signal
stop = None
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
    
    if args.outtoken == None: args.outtoken = args.runnr
    
    filenamelist = glob( "{}*run{}*gz".format(args.indir,args.runnr ))
    if len(filenamelist) == 0: 
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    
    fit = FitGammaLikelihood()
    fit.set_atmosphere("atmprof_paranal.dat")
    fit.set_instrument_description( *load_hessio(filenamelist[0]) )
    
    signal.signal(signal.SIGINT, signal_handler)
    for filename in sorted(filenamelist):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    # for now use only identical telescopes...
                                    allowed_tels=[1,2,3,4],
                                    #allowed_tels=TelDict[args.teltype],
                                    max_events=args.max_events)

        for event in source:
            ## TODO replace with actual pixel direction when they become available
            #for tel_id in event.dl0.tels_with_data:
                #if tel_id not in fit.pix_dirs:
                    #geom = CameraGeometry.guess(fit.cameras(tel_id)['PixX'].to(u.m), fit.cameras(tel_id)['PixY'].to(u.m), fit.telescopes['FL'][tel_id-1] * u.m)
                    #fit.pix_dirs[tel_id] = guessPixDirectionFocLength(geom.pix_x, geom.pix_y, tel_phi, tel_theta, fit.telescopes['FL'][tel_id-1] * u.m)


            print('Scanning input file... count = {}'.format(event.count))
            print('available telscopes: {}'.format(event.dl0.tels_with_data))

            fit.fill_pdf(event=event)
            
            
            if stop: break
        if stop: break
    
    print("writing file now")
    fit.write_raw("pdf/{}_{}".format(args.teltype,args.outtoken))
