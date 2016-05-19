from glob import glob
import argparse
from ctapipe.io.hessio import hessio_event_source
from FitGammaLikelihood import FitGammaLikelihood

from Telescope_Mask import TelDict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max-events', type=int, default=10000)
    parser.add_argument('-i', '--indir',   type=str, 
                        default="/local/home/tmichael/software/corsika_simtelarray/Data/sim_telarray/cta-ultra6/0.0deg/Data/gamma_20deg_180deg_run")
    parser.add_argument('-r', '--token',   type=str, default="20")
    parser.add_argument('-t', '--teltype', type=str, default="LST")
    args = parser.parse_args()
    
    filenamelist = glob( "{}*{}*gz".format(args.indir,args.token ))
    
    source = hessio_event_source(filenamelist[0],
                                 # for now use only identical telescopes...
                                 allowed_tels=TelDict[args.teltype],
                                 max_events=args.max_events-1)
    """
    Energy  = 0 * u.eV    # MC energy of the shower
    d       = 0 * u.m     # distance of the telescope to the shower's core
    delta   = 0 * u.rad   # angle between the pixel direction and the shower direction 
    rho     = 0 * u.rad   # angle between the pixel direction and the direction to the interaction vertex
    gamma   = 0 * u.rad   # angle between shower direction and the connecting vector between pixel-direction and vertex-direction
    npe_p_a = 0 * u.m**-2 # number of photo electrons generated per PMT area
    """
    
    fit = FitGammaLikelihood()
    
    for event in source:

        print('Scanning input file... count = {}'.format(event.count))
        print('available telscopes: {}'.format(event.dl0.tels_with_data))

        fit.fill_pdf(event=event)

    fit.write_raw("{}_{}_raw.npz".format(args.teltype, args.token))
    fit.write_pdf("{}_{}_pdf.npz".format(args.teltype, args.token))
