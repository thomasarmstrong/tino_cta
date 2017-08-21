import numpy as np
from astropy import units as u

from ctapipe.calib import CameraCalibrator

from ctapipe.image import hillas

from ctapipe.utils.linalg import rotation_matrix_2d

from modules.ImageCleaning import ImageCleaner, EdgeEventException
from ctapipe.utils.CutFlow import CutFlow

from collections import namedtuple, OrderedDict

PreparedEvent = namedtuple("PreparedEvent",
                           ["event", "hillas_dict", "n_tels",
                            "tot_signal", "max_signals",
                            "pos_fit", "dir_fit",
                            "err_est_pos", "err_est_dir"
                            ])

tel_phi = {}
tel_theta = {}
tel_orientation = (tel_phi, tel_theta)

# use this in the selection of the gain channels
np_true_false = np.array([[True], [False]])


def raise_error(message):
    raise ValueError(message)


class EventPreparator():
    def __init__(self, calib=None, cleaner=None, hillas_parameters=None,
                 shower_reco=None, event_cutflow=None, image_cutflow=None,
                 # event/image cuts:
                 allowed_cam_ids=None, min_ntel=1, min_charge=0, min_pixel=2):
        self.calib = calib or CameraCalibrator(None, None)
        self.cleaner = cleaner or ImageCleaner(mode=None)
        self.hillas_parameters = hillas_parameters or hillas.hillas_parameters
        self.shower_reco = shower_reco or \
            raise_error("need to provide a shower reconstructor....")

        # adding cutflows and cuts for events and images
        self.event_cutflow = event_cutflow or CutFlow("EventCutFlow")
        self.image_cutflow = image_cutflow or CutFlow("ImageCutFlow")

        self.event_cutflow.set_cuts(OrderedDict([
                    ("noCuts", None),
                    ("min2Tels trig", lambda x: x < min_ntel),
                    ("min2Tels reco", lambda x: x < min_ntel),
                    ("position nan", lambda x: np.isnan(x.value).any()),
                    ("direction nan", lambda x: np.isnan(x.value).any())
                ]))

        allowed_cam_ids = allowed_cam_ids or []
        self.image_cutflow.set_cuts(OrderedDict([
                    ("noCuts", None),
                    ("camera type", lambda id: allowed_cam_ids and
                                               id not in allowed_cam_ids),
                    ("min pixel", lambda s: np.count_nonzero(s) < min_pixel),
                    ("min charge", lambda x: x < min_charge),
                    ("poor moments", lambda m: m.width <= 0 or m.length <= 0)
                ]))

    def prepare_event(self, source):

        for event in source:

            self.event_cutflow.count("noCuts")

            if self.event_cutflow.cut("min2Tels trig", len(event.dl0.tels_with_data)):
                continue

            # calibrate the event
            self.calib.calibrate(event)

            # telescope loop
            tot_signal = 0
            max_signals = {}
            hillas_dict = {}
            n_tels = {"tot": len(event.dl0.tels_with_data),
                      "LST": 0, "MST": 0, "SST": 0}
            for tel_id in event.dl0.tels_with_data:
                self.image_cutflow.count("noCuts")

                camera = event.inst.subarray.tel[tel_id].camera

                # can this be improved?
                if tel_id not in tel_phi:
                    tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                    tel_theta[tel_id] = (np.pi/2-event.mc.tel[tel_id].altitude_raw)*u.rad

                # count the current telescope according to its size
                tel_type = event.inst.subarray.tel[tel_id].optics.tel_type
                n_tels[tel_type] += 1

                # temporary hack to have `FlashCam`s counted `n_tels`
                # but not used during the reconstruction
                if self.image_cutflow.cut("camera type", camera.cam_id):
                    continue

                # the camera image as a 1D array
                pmt_signal = event.dl1.tel[tel_id].image

                # the PMTs on some (most?) cameras have 2 gain channels. select one
                # according to a threshold (14 here). ultimately, this will be done IN the
                # camera/telescope itself but until then, do it here
                if pmt_signal.shape[0] > 1:
                    pick = (pmt_signal > 14).any(axis=0) != np_true_false
                    pmt_signal = pmt_signal.T[pick.T]
                else:
                    pmt_signal = np.squeeze(pmt_signal)

                # clean the image
                try:
                    pmt_signal, new_geom = \
                        self.cleaner.clean(pmt_signal.copy(), camera)

                    if self.image_cutflow.cut("min pixel", pmt_signal) or \
                       self.image_cutflow.cut("min charge", np.sum(pmt_signal)):
                        continue

                except FileNotFoundError as e:
                    print(e)
                    continue
                except EdgeEventException:
                    continue

                # could this go into `hillas_parameters` ...?
                max_signals[tel_id] = np.max(pmt_signal)

                # do the hillas reconstruction of the images
                try:
                    moments = self.hillas_parameters(new_geom.pix_x,
                                                     new_geom.pix_y, pmt_signal)

                    # if width and/or length are zero (e.g. when there is only only one
                    # pixel or when all  pixel are exactly in one row), the
                    # parametrisation won't be very useful: skip
                    if self.image_cutflow.cut("poor moments", moments):
                        continue

                except hillas.HillasParameterizationError as e:
                    print(e)
                    print("ignoring this camera")
                    continue

                hillas_dict[tel_id] = moments
                tot_signal += moments.size

            n_tels["reco"] = len(hillas_dict)
            if self.event_cutflow.cut("min2Tels reco", n_tels["reco"]):
                continue

            try:
                # telescope loop done, now do the core fit
                self.shower_reco.get_great_circles(hillas_dict, event.inst.subarray,
                                                   tel_phi, tel_theta)
                pos_fit, err_est_pos = self.shower_reco.fit_core_crosses()
                dir_fit, crossings = self.shower_reco.fit_origin_crosses()

                # don't have a direction error estimate yet
                err_est_dir = 0*u.deg
            except Exception as e:
                print(e)
                continue

            if self.event_cutflow.cut("position nan", pos_fit) or \
               self.event_cutflow.cut("direction nan", dir_fit):
                continue

            yield PreparedEvent(event=event, hillas_dict=hillas_dict, n_tels=n_tels,
                                tot_signal=tot_signal, max_signals=max_signals,
                                pos_fit=pos_fit, dir_fit=dir_fit,
                                err_est_pos=err_est_pos, err_est_dir=err_est_dir
                                )
