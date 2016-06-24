from sys import exit
from glob import glob
import argparse

from itertools import chain

import numpy as np

from astropy import units as u

from ctapipe.io import CameraGeometry
from ctapipe.io.hessio import hessio_event_source
from ctapipe.instrument.InstrumentDescription import load_hessio

from ctapipe.utils import linalg

from ctapipe.reco.hillas import hillas_parameters

from Telescope_Mask import TelDict

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

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
                        default="/local/home/tmichael/Data_b/cta/ASTRI9")
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('-t', '--teltype', type=str, default="SST_ASTRI")
    parser.add_argument('-o', '--outtoken', type=str, default=None)
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


    (h_telescopes, h_cameras, h_optics) = load_hessio(filenamelist_gamma[0])
    Ver = 'Feb2016'
    TelVer = 'TelescopeTable_Version{}'.format(Ver)
    CamVer = 'CameraTable_Version{}_TelID'.format(Ver)
    OptVer = 'OpticsTable_Version{}_TelID'.format(Ver)
    
    telescopes = h_telescopes[TelVer]
    cameras    = lambda tel_id : h_cameras[CamVer+str(tel_id)]
    optics     = lambda tel_id : h_optics [OptVer+str(tel_id)]

    tel_phi   =   0.*u.deg
    tel_theta =  20.*u.deg
    
    tel_geom = {}


    Features_g = []
    Class_g    = []
    Features_p = []
    Class_p    = []


    for filename in chain(filenamelist_gamma[:], filenamelist_proton[:]):
        print("filename = {}".format(filename))
        
        source = hessio_event_source(filename,
                                    # for now use only identical telescopes...
                                    #allowed_tels=TelDict[args.teltype],
                                    max_events=args.max_events)
        
        
        
        
        for event in source:
            mc_shower = event.mc
            mc_shower_core = np.array( [mc_shower.core_x.value, mc_shower.core_y.value] )
            
            tel_data = {}
            for tel_id, tel in event.mc.tel.items():
                tel_data[tel_id] = tel.photo_electrons
            
            sizes   = []
            widths  = []
            for tel_id, photo_electrons in tel_data.items():

                tel_idx = np.searchsorted( telescopes['TelID'], tel_id )
                tel_pos = np.array( [telescopes["TelX"][tel_idx], telescopes["TelY"][tel_idx]] )
                
                photo_electrons = np.array(photo_electrons * sum((tel_pos - mc_shower_core)**2), dtype=int)
                
                if tel_id not in tel_geom:
                    tel_geom[tel_id] = CameraGeometry.guess(cameras(tel_id)['PixX'].to(u.m),
                                                            cameras(tel_id)['PixY'].to(u.m),
                                                            telescopes['FL'][tel_idx] * u.m) 
                moments = hillas_parameters(tel_geom[tel_id].pix_x,
                                            tel_geom[tel_id].pix_y,
                                            photo_electrons)
                
                
                sizes  .append(moments.size)
                widths .append(moments.width.value if moments.width.value==moments.width.value else 0)
                
            widths = np.array(widths)
            mean_size   = sum(sizes)   / len(sizes)
            mean_width  = sum(widths)  / len(widths) 
            RMS_width   = np.mean( ( widths  - mean_width )**2 )**.5

            if filename in filenamelist_proton:
                Features_p.append( [mean_size, mean_width, RMS_width, mc_shower.energy.value] )
                Class_p.append( "p" )
            else:
                Features_g.append( [mean_size, mean_width, RMS_width, mc_shower.energy.value] )
                Class_g.append( "g" )
        if stop: break
    
    
    
    Features = Features_p + Features_g
    Class    = Class_p    + Class_g   
    wrong_p = 0
    total_p = 0
    wrong_g = 0
    total_g = 0
    
    for split, bla in enumerate(Features):
            
        tFeatures = Features[:split] 
        tClass    = Class[:split]
        if split < len(Features):
            tFeatures += Features[split+1:]
            tClass    += Class   [split+1:]

        #clf = svm.SVC(kernel='rbf')
        clf = RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=1, random_state=0)
        clf.fit(tFeatures, tClass)
    
        predict = clf.predict([Features[split]])
        
        if Class[split] == "p":
            total_p += 1
            if Class[split] != predict: wrong_p += 1
        else:
            total_g += 1
            if Class[split] != predict: wrong_g += 1
        
        if total_p:
            print( "wrong p: {} out of {} => {}".format(wrong_p, total_p,wrong_p / total_p *100*u.percent))
        if total_g:
            print( "wrong g: {} out of {} => {}".format(wrong_g, total_g,wrong_g / total_g *100*u.percent))
        print()
        if stop: break

    if total_p:
        print( "wrong p: {} out of {} => {}".format(wrong_p, total_p,wrong_p / total_p *100*u.percent))
    if total_g:
        print( "wrong g: {} out of {} => {}".format(wrong_g, total_g,wrong_g / total_g *100*u.percent))