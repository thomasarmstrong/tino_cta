from sys import exit

from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

from ctapipe.io.camera import CameraGeometry
from ctapipe.io.hessio import hessio_event_source

from ctapipe.utils import linalg

from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters_4 as hillas_parameters

from helper_functions import *
from modules.CutFlow import CutFlow
from modules.EventClassifier import *
from modules.ImageCleaning import ImageCleaner, \
                                  EdgeEventException, UnknownModeException

from ctapipe.reco.FitGammaHillas import \
    FitGammaHillas, TooFewTelescopesException


# PyTables
try:
    import tables as tb
except:
    print("no pytables installed")


def main():

    # your favourite units here
    angle_unit  = u.deg
    energy_unit = u.GeV
    dist_unit   = u.m

    agree_threshold = .5
    min_tel = 3

    parser = make_argparser()
    parser.add_argument('--classifier', type=str,
                        default='data/classifier_pickle/classifier_{}_{}_{}.pkl')
    parser.add_argument('-o', '--out_file', type=str,
                        default="data/reconstructed_events/classified_events_{}_{}.h5",
                        help="location to write the classified events to. placeholders "
                             "are meant as {particle type} and {cleaning mode}")
    parser.add_argument('--proton',  action='store_true',
                        help="do protons instead of gammas")
    parser.add_argument('--wave_dir',  type=str, default=None,
                        help="directory where to find mr_filter. "
                             "if not set look in $PATH")
    parser.add_argument('--wave_temp_dir',  type=str, default='/tmp/',
                        help="directory where mr_filter to store the temporary fits files"
                        )

    args = parser.parse_args()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    elif args.proton:
        filenamelist = sorted(glob("{}/proton/*gz".format(args.indir)))[100:]
    else:
        filenamelist = sorted(glob("{}/gamma/*gz".format(args.indir)))[14:]

    if len(filenamelist) == 0:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}
    tel_orientation = (tel_phi, tel_theta)

    # counting events and where they might have gone missing
    Eventcutflow = CutFlow("EventCutFlow")
    Imagecutflow = CutFlow("ImageCutFlow")
    Eventcutflow.set_cut("noCuts", None)
    Eventcutflow.set_cut("min2Tels trig", lambda x: x < 2)
    Eventcutflow.set_cut("min2Tels reco", lambda x: x < 2)

    # class that wraps tail cuts and wavelet cleaning
    Cleaner = ImageCleaner(mode=args.mode, wavelet_options=args.raw,
                           tmp_files_directory=args.wave_temp_dir,
                           mrfilter_directory=args.wave_dir,
                           skip_edge_events=False)  # args.skip_edge_events)

    # simple hillas-based shower reco
    fit = FitGammaHillas()

    # wrapper for the scikit learn classifier
    classifier = fancy_EventClassifier.load(
                    args.classifier.format(args.mode,
                                           args.raw.replace(' ', '').replace(',', ''),
                                           "RandomForestClassifier"))

    # catch ctr-c signal to exit current loop and still display results
    signal_handler = SignalHandler()
    signal.signal(signal.SIGINT, signal_handler)

    # this class defines the reconstruction parameters to keep track of
    class RecoEvent(tb.IsDescription):
        NTels_trig = tb.Int16Col(dflt=1, pos=0)
        NTels_reco = tb.Int16Col(dflt=1, pos=1)
        MC_Energy = tb.Float32Col(dflt=1, pos=2)
        phi = tb.Float32Col(dflt=1, pos=3)
        theta = tb.Float32Col(dflt=1, pos=4)
        off_angle = tb.Float32Col(dflt=1, pos=5)
        ErrEstPos = tb.Float32Col(dflt=1, pos=6)
        gammaness = tb.Float32Col(dflt=1, pos=9)

    channel = "gamma" if "gamma" in " ".join(filenamelist) else "proton"
    reco_outfile = tb.open_file(
            # trying to put particle type and cleaning mode into the filename
            # `format` puts in each argument as long as there is a free "{}" token
            # if `out_file` was set without any "{}", nothing will be replaced
            args.out_file.format(channel, args.mode), mode="w",
            # if we don't want to write the event list to disk, need to add more arguments
            **({} if args.store else {"driver": "H5FD_CORE",
                                      "driver_core_backing_store": False}))
    reco_table = reco_outfile.create_table("/", "reco_events", RecoEvent)
    reco_event = reco_table.row

    source_orig = None

    allowed_tels = range(10)  # smallest ASTRI array
    # allowed_tels = range(34)  # all ASTRI telescopes
    for filename in filenamelist[:args.last]:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                        allowed_tels=allowed_tels,
                                        max_events=args.max_events)

        # event loop
        for event in source:

            Eventcutflow.count("noCuts")

            mc_shower = event.mc
            mc_shower_core = np.array([mc_shower.core_x.value,
                                       mc_shower.core_y.value]) * u.m

            if Eventcutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                continue

            # telescope loop
            tot_signal = 0
            max_signal = 0
            hillas_dict = {}
            for tel_id in event.dl0.tels_with_data:
                Imagecutflow.count("noCuts")

                if source_orig is None:
                    source_orig = linalg.set_phi_theta(
                        event.mc.tel[tel_id].azimuth_raw * u.rad,
                        (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad)

                pmt_signal = apply_mc_calibration_ASTRI(
                                event.r0.tel[tel_id].adc_sums,
                                event.mc.tel[tel_id].dc_to_pe,
                                event.mc.tel[tel_id].pedestal)

                max_signal = np.max(pmt_signal)

                # guessing camera geometry
                if tel_id not in cam_geom:
                    cam_geom[tel_id] = CameraGeometry.guess(
                                        event.inst.pixel_pos[tel_id][0],
                                        event.inst.pixel_pos[tel_id][1],
                                        event.inst.optical_foclen[tel_id])
                    tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                    tel_theta[tel_id] = (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad

                # trying to clean the image
                try:
                    pmt_signal, new_geom = \
                        Cleaner.clean(pmt_signal.copy(), cam_geom[tel_id],
                                      event.inst.optical_foclen[tel_id])

                    if np.count_nonzero(pmt_signal) < 3:
                        continue

                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEventException:
                    continue

                Imagecutflow.count("cleaning")

                # trying to do the hillas reconstruction of the images
                try:
                    moments = hillas_parameters(new_geom.pix_x,
                                                new_geom.pix_y,
                                                pmt_signal)

                    if not (moments.width > 0 and moments.length > 0):
                        continue

                    if False:
                        from ctapipe.visualization import CameraDisplay
                        fig = plt.figure(figsize=(17, 10))

                        ax1 = fig.add_subplot(121)
                        disp1 = CameraDisplay(cam_geom[tel_id],
                                              image=np.sqrt(event.mc.tel[tel_id]
                                                            .photo_electron_image),
                                              ax=ax1)
                        disp1.cmap = plt.cm.inferno
                        disp1.add_colorbar()
                        plt.title("sqrt photo-electron image")

                        ax3 = fig.add_subplot(122)
                        disp3 = CameraDisplay(new_geom,
                                              image=np.sqrt(pmt_signal),
                                              ax=ax3)
                        disp3.overlay_moments(moments, color='seagreen', linewidth=3)
                        disp3.cmap = plt.cm.inferno
                        disp3.add_colorbar()
                        plt.title("sqrt cleaned image")
                        plt.suptitle(args.mode)
                        plt.show()

                except HillasParameterizationError as e:
                    print(e)
                    print("ignoring this camera")
                    continue

                hillas_dict[tel_id] = moments
                tot_signal += moments.size

                Imagecutflow.count("Hillas")

            if Eventcutflow.cut("min2Tels reco", len(hillas_dict)):
                continue

            try:
                # telescope loop done, now do the core fit
                fit.get_great_circles(hillas_dict,
                                      event.inst,
                                      *tel_orientation)
                pos_fit_cr, err_est_pos = fit.fit_core_crosses()
            except Exception as e:
                print(e)
                continue

            if np.isnan(pos_fit_cr.value).any():
                continue

            pos_fit = pos_fit_cr

            Eventcutflow.count("position fit")

            # now prepare the features for the classifier
            features_evt = {}
            NTels = len(hillas_dict)
            for tel_id in hillas_dict.keys():
                Imagecutflow.count("pre-features")

                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                moments = hillas_dict[tel_id]

                impact_dist_sim = linalg.length(tel_pos-mc_shower_core)
                impact_dist_rec = linalg.length(tel_pos-pos_fit)
                features_tel = [
                            impact_dist_rec/u.m,
                            tot_signal,
                            max_signal,
                            moments.size,
                            NTels,
                            moments.width/u.m,
                            moments.length/u.m,
                            moments.skewness,
                            moments.kurtosis,
                            err_est_pos/u.m
                            ]

                if np.isnan(features_tel).any():
                    continue

                Imagecutflow.count("features nan")

                features_evt[tel_id] = features_tel

            if not features_evt:
                continue

            predict_proba = classifier.predict_proba([features_evt])
            gammaness = predict_proba[0, 0]

            fit_dir, crossings = fit.fit_origin_crosses()

            off_angle = linalg.angle(fit_dir, source_orig)
            phi, theta = linalg.get_phi_theta(fit_dir)
            phi = (phi if phi > 0 else phi+360*u.deg)

            reco_event["NTels_trig"] = len(event.dl0.tels_with_data)
            reco_event["NTels_reco"] = NTels
            reco_event["MC_Energy"] = event.mc.energy / energy_unit
            reco_event["phi"] = phi / angle_unit
            reco_event["theta"] = theta / angle_unit
            reco_event["off_angle"] = off_angle / angle_unit
            reco_event["ErrEstPos"] = err_est_pos / dist_unit
            reco_event["gammaness"] = gammaness
            reco_event.append()
            reco_table.flush()

            if signal_handler.stop:
                break
        if signal_handler.stop:
            break

    Eventcutflow()
    Imagecutflow()

    if args.plot:
        gammaness = [x['gammaness'] for x in reco_table]
        NTels_rec = [x['NTels_reco'] for x in reco_table]

        fig = plt.figure()
        ax = plt.subplot(111)
        histo = np.histogram2d(NTels_rec, gammaness,
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        im = ax.imshow(histo_normed, interpolation='none', origin='lower',
                       aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
        cb = fig.colorbar(im, ax=ax)
        ax.set_xlabel("NTels")
        ax.set_ylabel("drifted gammaness")
        plt.title(" ** ".join([args.mode, "protons" if args.proton else "gamma"]))

    N_selected = len([x for x in reco_table.where(
        """(NTels_reco > min_tel) & (gammaness > agree_threshold)""")])
    N_total = len(reco_table)
    print("\nfraction selected events:")
    print("{} / {} = {} %".format(N_selected, N_total, N_selected/N_total*100))

    print("\nlength filenamelist:", len(filenamelist[:args.last]))

    plt.show()


if __name__ == '__main__':
    main()
