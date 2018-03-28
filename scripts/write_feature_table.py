#!/usr/bin/env python

from tino_cta.helper_functions import *
from astropy import units as u

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

from ctapipe.reco.HillasReconstructor import HillasReconstructor

# tino_cta
from tino_cta.ImageCleaning import ImageCleaner
from tino_cta.prepare_event import EventPreparer


if __name__ == "__main__":

    # your favourite units here
    energy_unit = u.TeV
    angle_unit = u.deg
    dist_unit = u.m

    parser = make_argparser()
    parser.add_argument('-o', '--outfile', type=str, required=True)

    parser.add_argument('--wave_dir', type=str, default=None,
                        help="directory where to find mr_filter. "
                             "if not set look in $PATH")
    parser.add_argument('--wave_temp_dir', type=str, default='/dev/shm/',
                        help="directory where mr_filter to store the temporary fits"
                             " files")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gamma', default=True, action='store_true',
                       help="do gammas (default)")
    group.add_argument('--proton', action='store_true',
                       help="do protons instead of gammas")
    group.add_argument('--electron', action='store_true',
                       help="do electrons instead of gammas")

    args = parser.parse_args()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    elif args.proton:
        filenamelist = glob("{}/proton/*gz".format(args.indir))
        channel = "proton"
    elif args.electron:
        filenamelist = glob("{}/electron/*gz".format(args.indir))
        channel = "electron"
    elif args.gamma:
        filenamelist = glob("{}/gamma/*gz".format(args.indir))
        channel = "gamma"
    else:
        raise ValueError("don't know which input to use...")

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)
    else:
        print("found {} files".format(len(filenamelist)))

    # keeping track of events and where they were rejected
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")

    # takes care of image cleaning
    cleaner = ImageCleaner(mode=args.mode, cutflow=Imagecutflow,
                           wavelet_options=args.raw,
                           tmp_files_directory=args.wave_temp_dir,
                           skip_edge_events=True, island_cleaning=True)

    # the class that does the shower reconstruction
    shower_reco = HillasReconstructor()

    preper = EventPreparer(
        cleaner=cleaner,
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
        MC_Energy = tb.FloatCol(dflt=1, pos=14)

    feature_outfile = tb.open_file(args.outfile, mode="w")
    feature_table = {}
    feature_events = {}

    pe_thersh = 100
    n_faint_img = []
    n_total_img = []
    mc_energy = []

    allowed_tels = set(prod3b_tel_ids("L+N+D"))
    for i, filename in enumerate(filenamelist[:50][:args.last]):
        print("file: {} filename = {}".format(i, filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        # loop that cleans and parametrises the images and performs the reconstruction
        for (event, hillas_dict, n_tels,
             tot_signal, max_signals, pos_fit, dir_fit, h_max,
             err_est_pos, err_est_dir) in preper.prepare_event(source):

            n_faint = 0
            for tel_id in hillas_dict.keys():
                cam_id = event.inst.subarray.tel[tel_id].camera.cam_id

                if cam_id not in feature_events:
                    feature_table[cam_id] = feature_outfile.create_table(
                        '/', '_'.join(["feature_events", cam_id]), EventFeatures)
                    feature_events[cam_id] = feature_table[cam_id].row

                moments = hillas_dict[tel_id]
                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m
                impact_dist = linalg.length(tel_pos - pos_fit)

                if moments.size > pe_thersh:
                    n_faint += 1

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

            n_faint_img.append(n_faint)
            n_total_img.append(len(hillas_dict))
            mc_energy.append(event.mc.energy / energy_unit)

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break

    # make sure that all the events are properly stored
    for table in feature_table.values():
        table.flush()

    # def averages(values, bin_values, bin_edges):
    #     averages_binned = \
    #         np.squeeze(np.full((len(bin_edges) - 1, len(values.shape)), np.inf))
    #     for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #         try:
    #             averages_binned[i] = \
    #                 np.mean(values[(bin_values > bin_l) & (bin_values < bin_h)])
    #         except IndexError:
    #             pass
    #     return averages_binned.T

    # energy_bin_edges = np.logspace(-2.1, 2.5, 24)
    # faint_img_fraction = np.array(n_faint_img) / np.array(n_total_img)
    # faint_img_fraction_averages = averages(faint_img_fraction, mc_energy,
    #                                        energy_bin_edges)
    # plt.figure()
    # plt.semilogx(np.sqrt(energy_bin_edges[1:] * energy_bin_edges[:-1]),
    #              faint_img_fraction_averages)
    # plt.xlabel(r'$E_\mathrm{reco}$' + ' / {:latex}'.format(energy_unit))
    # plt.ylabel("fraction of faint imgages (pe < 100) per event")
    # save_fig("plots/faint_img_fraction_{}_{}".format(args.mode, channel))
#    plt.show()
