#!/usr/bin/env python3

from helper_functions import *
from astropy import units as u
from collections import namedtuple

from sys import exit, path
from os.path import expandvars
from glob import glob
from itertools import chain

# PyTables
import tables as tb

# ctapipe
from ctapipe.utils import linalg
from ctapipe.utils.CutFlow import CutFlow

from ctapipe.instrument import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from ctapipe.reco.HillasReconstructor import \
    HillasReconstructor, TooFewTelescopesException

# tino_cta
from modules.ImageCleaning import *
from modules.prepare_event import EventPreparator


if __name__ == "__main__":

    # your favourite units here
    energy_unit = u.TeV
    angle_unit = u.deg
    dist_unit = u.m

    parser = make_argparser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    filenamelist = sorted(glob("{}/*gz".format(args.indir)))
    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           skip_edge_events=False, island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    preper = EventPreparator(cleaner=cleaner,
                             hillas_parameters=hillas_parameters,
                             shower_reco=shower_reco,
                             event_cutflow=Eventcutflow, image_cutflow=Imagecutflow,
                             # event/image cuts:
                             allowed_cam_ids=[],
                             min_ntel=2,
                             min_charge=args.min_charge, min_pixel=3)

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    class EventFeatures(tb.IsDescription):
        impact_dist = tb.Float32Col(dflt=1, pos=0)
        sum_signal_evt = tb.Float32Col(dflt=1, pos=1)
        max_signal_cam = tb.Float32Col(dflt=1, pos=2)
        sum_signal_cam = tb.Float32Col(dflt=1, pos=3)
        N_LST = tb.Int16Col(dflt=1, pos=4)
        N_MST = tb.Int16Col(dflt=1, pos=5)
        N_SST = tb.Int16Col(dflt=1, pos=6)
        width = tb.Float32Col(dflt=1, pos=7)
        length = tb.Float32Col(dflt=1, pos=8)
        skewness = tb.Float32Col(dflt=1, pos=9)
        kurtosis = tb.Float32Col(dflt=1, pos=10)
        h_max = tb.Float32Col(dflt=1, pos=11)
        err_est_pos = tb.Float32Col(dflt=1, pos=12)
        err_est_dir = tb.Float32Col(dflt=1, pos=13)
        MC_energy = tb.FloatCol(dflt=1, pos=14)

    feature_outfile = tb.open_file(args.outfile, mode="w")
    feature_table_lst = feature_outfile.create_table("/", "feature_events_lst",
                                                     EventFeatures)
    feature_table_nec = feature_outfile.create_table("/", "feature_events_nec",
                                                     EventFeatures)
    feature_table_dig = feature_outfile.create_table("/", "feature_events_dig",
                                                     EventFeatures)
    feature_events = {}
    feature_events["LSTCam"] = feature_table_lst.row
    feature_events["NectarCam"] = feature_table_nec.row
    feature_events["DigiCam"] = feature_table_dig.row

    allowed_tels = prod3b_tel_ids("L+N+D")
    for filename in filenamelist[:50][:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # loop that cleans and parametrises the images and performs the reconstruction
        for (event, hillas_dict, n_tels,
             tot_signal, max_signals, pos_fit, dir_fit, h_max,
             err_est_pos, err_est_dir) in preper.prepare_event(source):

            for tel_id in hillas_dict.keys():
                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id
                moments = hillas_dict[tel_id]
                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m
                impact_dist = linalg.length(tel_pos - pos_fit)

                feature_events[cam_id]["impact_dist"] = impact_dist / dist_unit
                feature_events[cam_id]["sum_signal_evt"] = tot_signal
                feature_events[cam_id]["max_signal_cam"] = max_signals[tel_id]
                feature_events[cam_id]["sum_signal_cam"] = moments.size
                feature_events[cam_id]["N_LST"] = n_tels["LST"]
                feature_events[cam_id]["N_MST"] = n_tels["MST"]
                feature_events[cam_id]["N_SST"] = n_tels["SST"]
                feature_events[cam_id]["width"] = moments.width / dist_unit
                feature_events[cam_id]["length"] = moments.length / dist_unit
                feature_events[cam_id]["skewness"] = moments.skewness
                feature_events[cam_id]["kurtosis"] = moments.kurtosis
                feature_events[cam_id]["h_max"] = h_max / dist_unit
                feature_events[cam_id]["err_est_pos"] = err_est_pos / dist_unit
                feature_events[cam_id]["err_est_dir"] = err_est_dir / angle_unit
                feature_events[cam_id]["MC_Energy"] = event.mc.energy / energy_unit
                feature_events[cam_id].append()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break
