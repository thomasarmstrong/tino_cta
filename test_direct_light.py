#!/usr/bin/env python

from glob import glob
import warnings

from astropy import units as u

import matplotlib.pyplot as plt

from ctapipe.io.hessio import hessio_event_source

from ctapipe.calib import CameraCalibrator

from ctapipe.utils import linalg

from ctapipe.visualization import CameraDisplay

from helper_functions import *

from modules.ImageCleaning import ImageCleaner, EdgeEvent

from IPython import embed

import matplotlib.animation as animation


# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])
pe_thresh = {
    "ASTRICam": 14,
    "LSTCam": 100,
    "NectarCam": 190}

if __name__ == "__main__":

    parser = make_argparser()
    parser.add_argument('--type', default="helium")

    args = parser.parse_args()

    filenamelist = sorted(glob("{}/CR_{}/*gz".format(args.indir, args.type)))

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

    allowed_tels = None
    for filename in filenamelist:
        print("filename = {}".format(filename))

        source = hessio_event_source(filename,
                                     allowed_tels=allowed_tels,
                                     max_events=args.max_events)

        for event in source:

            # if event.mc.energy.to("TeV").value < 1:
            #     continue

            if (event.mc.alt.degree-70)**2 + (event.mc.az.degree-180)**2 > 3:
                continue

            print()
            print(event.mc.energy)

            DeltaAlt = (event.mc.alt)
            DeltaAz = (event.mc.az)
            print(DeltaAz.degree, DeltaAlt.degree)

            # calibrate the event
            calib.calibrate(event)

            for tel_id in event.dl0.tels_with_data:

                camera = event.inst.subarray.tel[tel_id].camera

                foclen = event.inst.subarray.tel[tel_id].optics.effective_focal_length

                shower_pos = np.array([event.mc.core_x/u.m, event.mc.core_y/u.m])*u.m
                tel_pos = np.array(event.inst.tel_pos[tel_id][:2]) * u.m

                print(linalg.length(tel_pos-shower_pos))

                pmt_simu = event.mc.tel[tel_id].photo_electron_image

                # embed()

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
                        pmt_wave, geom_wave = \
                            cleaner_wave.clean(pmt_signal.copy(), camera)
                        pmt_tail, geom_tail = \
                            cleaner_tail.clean(pmt_signal.copy(), camera)
                except:
                    continue

                fig = plt.figure(figsize=(20, 12))

                ax = fig.add_subplot(222)
                disp_uncl = CameraDisplay(camera, image=pmt_signal, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("noisy, calibr. image")
                disp_uncl.add_colorbar()

                ax = fig.add_subplot(221)
                disp_simu = CameraDisplay(camera, image=pmt_simu, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("photo-electron image")
                disp_simu.add_colorbar()

                ax = fig.add_subplot(223)
                disp_wave = CameraDisplay(geom_wave, image=pmt_wave, ax=ax)
                ax.set_aspect('equal', adjustable='box')
                plt.title("wave cleaned image")
                disp_wave.add_colorbar()

                ax = fig.add_subplot(224)
                # disp_tail = CameraDisplay(geom_tail, image=pmt_tail, ax=ax)
                # ax.set_aspect('equal', adjustable='box')
                # plt.title("tail cleaned image")
                # disp_tail.add_colorbar()
                #
                # fig2 = plt.figure()
                samples = event.dl0.tel[tel_id].pe_samples[0, :, :]
                disp_anim = CameraDisplay(camera, image=samples[:, 0])
                txt = ax.text(1, 1, "frame {}".format(0))

                def updatefig(frame_number):
                    global disp_anim, txt
                    index = frame_number % samples.shape[-1]
                    disp_anim.image = samples[:, index]
                    txt.set_text("frame {}".format(index))

                anim = animation.FuncAnimation(fig, updatefig, interval=50)
                plt.tight_layout()
                plt.tight_layout()
                # anim.save("plots/iron/event_1/{}.mp4".format(tel_id))
                plt.show()
