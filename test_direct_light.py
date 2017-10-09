#!/usr/bin/env python

from glob import glob
import warnings

import matplotlib.pyplot as plt

from ctapipe.io.hessio import hessio_event_source

from ctapipe.calib import CameraCalibrator

from ctapipe.visualization import CameraDisplay

from helper_functions import *

from modules.ImageCleaning import ImageCleaner, EdgeEvent


# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])
pe_thresh = {
    "ASTRICam": 14,
    "LSTCam": 100,
    "NectarCam": 190}

if __name__ == "__main__":

    parser = make_argparser()
    parser.add_argument('--gamma',  action='store_false', dest='proton', default=True,
                        help="do gammas instead of protons")

    args = parser.parse_args()

    if args.infile_list:
        filenamelist = []
        for f in args.infile_list:
            filenamelist += glob("{}/{}".format(args.indir, f))
        filenamelist.sort()
    elif args.proton:
        filenamelist = sorted(glob("{}/proton/*gz".format(args.indir)))
    else:
        filenamelist = sorted(glob("{}/gamma/*gz".format(args.indir)))

    if not filenamelist:
        print("no files found; check indir: {}".format(args.indir))
        exit(-1)

    calib = CameraCalibrator(None, None)

    # takes care of image cleaning
    cleaner_wave = ImageCleaner(mode="wave",
                                wavelet_options=args.raw,
                                skip_edge_events=False, island_cleaning=False)
    cleaner_tail = ImageCleaner(mode="tail",
                                wavelet_options=args.raw,
                                skip_edge_events=False, island_cleaning=False)

    allowed_tels = prod3b_tel_ids(args.cam_ids[0])
    for filename in filenamelist:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            if event.mc.energy.to("TeV").value < 1:
                continue
            print(event.mc.energy)

            # calibrate the event
            calib.calibrate(event)

            for tel_id in event.dl0.tels_with_data:

                camera = event.inst.subarray.tel[tel_id].camera
                # the camera image as a 1D array
                pmt_signal = event.dl1.tel[tel_id].image

                # the PMTs on some (most?) cameras have 2 gain channels. select one
                # according to a threshold. ultimately, this will be done IN the
                # camera/telescope itself but until then, do it here
                if pmt_signal.shape[0] > 1:
                    pmt_signal = np.squeeze(pmt_signal)
                    pick = (pe_thresh[camera.cam_id]
                            < pmt_signal).any(axis=0) != np_true_false
                    pmt_signal = pmt_signal.T[pick.T]
                else:
                    pmt_signal = np.squeeze(pmt_signal)

                # clean the image
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pmt_wave, geom_wave = cleaner_wave.clean(pmt_signal.copy(), camera)
                        pmt_tail, geom_tail = cleaner_tail.clean(pmt_signal.copy(), camera)
                except:
                    continue

                fig = plt.figure(figsize=(18, 12))

                ax = fig.add_subplot(221)
                disp_uncl = CameraDisplay(camera, image=pmt_signal, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("noisy, calibr. image")
                disp_uncl.add_colorbar()

                ax = fig.add_subplot(223)
                disp_wave = CameraDisplay(camera, image=pmt_wave, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("wave cleaned image")
                disp_wave.add_colorbar()

                ax = fig.add_subplot(224)
                disp_tail = CameraDisplay(camera, image=pmt_tail, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("tail cleaned image")
                disp_tail.add_colorbar()

                plt.show()
