import numpy as np

from copy import copy

from scipy import ndimage
from matplotlib import pyplot as plt

from ctapipe.image.cleaning import tailcuts_clean, dilate

from ctapipe.utils.CutFlow import CutFlow


try:
    from ctapipe.image.geometry_converter import convert_geometry_1d_to_2d, \
                                                 convert_geometry_back
except:
    print("something missing from ctapipe.image.geometry_converter -- numba?")

try:
    from datapipe.denoising.wavelets_mrfilter import WaveletTransform
    from datapipe.denoising import cdf
    from datapipe.denoising.inverse_transform_sampling import \
        EmpiricalDistribution as EmpDist

except:
    print("something missing from datapipe.denoising.wavelets_mrfilter -- skimage?")

from datapipe.io.geometry_converter import astri_to_2d_array, array_2d_to_astri


class UnknownMode(Exception):
    pass


class EdgeEvent(Exception):
    pass


class MissingImplementation(Exception):
    pass


def kill_isolpix(array, neighbours=None, threshold=.2):
    """
    Return array with isolated islands removed.
    Only keeping the biggest islands (largest surface).

    Parameters
    ----------
    array : 2D array
        the input image you want to keep the "biggest" patch of
    neighbours : 2D array, optional (default: None)
        a mask defining what is considered a neighbour
    threshold : float, optional (default: 0.2)
        ignores pixel with entries below this value

    Returns
    -------
    filtered_array : 2D array
        image with just the largest island remaining
    """

    filtered_array = np.copy(array)

    filtered_array[filtered_array < threshold] = 0
    mask = filtered_array > 0

    label_im, nb_labels = ndimage.label(mask, neighbours)

    sums = ndimage.sum(filtered_array, label_im, range(nb_labels + 1))
    mask_sum = sums < np.max(sums)
    remove_pixel = mask_sum[label_im]

    filtered_array[remove_pixel] = 0

    return filtered_array


def remove_plateau(img):
    img -= np.mean(img)
    img[img < 0] = 0


def raise_minimum(img):
    img -= np.min(img)


class ImageCleaner:

    hex_neighbours_1ring = np.array([[1, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 1]])
    hex_neighbours_2ring = np.array([[1, 1, 1, 0, 0],
                                     [1, 1, 1, 1, 0],
                                     [1, 1, 1, 1, 1],
                                     [0, 1, 1, 1, 1],
                                     [0, 0, 1, 1, 1]])

    def __init__(self, mode="wave", dilate=False, island_cleaning=True,
                 skip_edge_events=True, edge_width=1,
                 cutflow=CutFlow("ImageCleaner"),
                 wavelet_options=None,
                 tmp_files_directory='/dev/shm/', mrfilter_directory=None):
        self.mode = mode
        self.skip_edge_events = skip_edge_events
        self.edge_width = edge_width
        self.cutflow = cutflow

        if mode in [None, "none", "None"]:
            self.clean = self.clean_none
        elif mode == "wave":
            self.clean = self.clean_wave
            self.wavelet_cleaning = \
                lambda *arg, **kwargs: WaveletTransform().clean_image(
                                *arg, **kwargs,
                                kill_isolated_pixels=True,
                                tmp_files_directory=tmp_files_directory,
                                mrfilter_directory=mrfilter_directory)
            self.island_threshold = 1.5

            # command line parameters for the mr_filter call
            self.wavelet_options = \
                {"ASTRICam": wavelet_options or "-K -C1 -m3 -s2,2,3,3 -n4",
                 "FlashCam": wavelet_options or "-K -C1 -m3 -s4,4,5,4 -n4",
                 "LSTCam": wavelet_options or "-K -C1 -m3 -s2,4.5,3.5,3 -n4"}
            # parameters for poisson + gauß noise injection
            self.noise_model = \
                {"ASTRICam": EmpDist(cdf.ASTRI_CDF_FILE),
                 "FlashCam": EmpDist(cdf.FLASHCAM_CDF_FILE),
                 "LSTCam": EmpDist(cdf.LSTCAM_CDF_FILE)}

        elif mode == "tail":
            self.clean = self.clean_tail
            self.tail_thresholds = \
                {"ASTRICam": (5, 7),  # (5, 10)?
                 "FlashCam": (12, 15),
                 # ASWG Zeuthen talk by Abelardo Moralejo:
                 "LSTCam": (5, 10),
                 "NectarCam": (4, 8),
                 # "FlashCam": (4, 8),  # there is some scaling missing?
                 "SSTCam": (3, 6),
                 "GCTCam": (2, 4),
                 "SCTCam": (1.5, 3)}

            self.island_threshold = 1.5
            self.dilate = dilate
        else:
            raise UnknownMode(
                'cleaning mode "{}" not found'.format(mode))

        if island_cleaning:
            self.island_cleaning = kill_isolpix
        else:
            # just a pass-through that does nothing
            self.island_cleaning = lambda x, *args, **kw: x

    def clean_wave(self, img, cam_geom):
        if cam_geom.pix_type.startswith("hex"):
            return self.clean_wave_hex(img, cam_geom)

        elif "ASTRI" in cam_geom.cam_id:
            return self.clean_wave_astri(img, cam_geom)

        else:
            raise MissingImplementation(
                    "wavelet cleaning of square-pixel images only for ASTRI so far")

    def clean_wave_astri(self, img, cam_geom):
        array2d_img = astri_to_2d_array(img)
        cleaned_img = self.wavelet_cleaning(
                array2d_img, raw_option_string=self.wavelet_options[cam_geom.cam_id],
                noise_distribution=self.noise_model[cam_geom.cam_id])

        self.cutflow.count("wavelet cleaning")

        # wavelet_transform still leaves some isolated pixels; remove them
        cleaned_img = self.island_cleaning(cleaned_img)

        if self.skip_edge_events:
            edge_thresh = np.max(cleaned_img)/5.
            mask = np.ones_like(cleaned_img, bool)
            mask[self.edge_width:-self.edge_width, self.edge_width:-self.edge_width] = 0
            if (cleaned_img[mask] > edge_thresh).any():
                    raise EdgeEvent
            self.cutflow.count("wavelet edge")

        new_img = array_2d_to_astri(cleaned_img)
        new_geom = cam_geom

        return new_img, new_geom

    def clean_wave_hex(self, img, cam_geom):
        rot_geom, rot_img = convert_geometry_1d_to_2d(
                                cam_geom, img, cam_geom.cam_id)

        cleaned_img = self.wavelet_cleaning(
                rot_img, raw_option_string=self.wavelet_options[cam_geom.cam_id],
                noise_distribution=self.noise_model[cam_geom.cam_id])

        self.cutflow.count("wavelet cleaning")

        cleaned_img = self.island_cleaning(cleaned_img,
                                           neighbours=self.hex_neighbours_1ring,
                                           threshold=self.island_threshold)

        unrot_geom, unrot_img = convert_geometry_back(rot_geom, cleaned_img,
                                                      cam_geom.cam_id)

        new_img = unrot_img
        new_geom = unrot_geom

        return new_img, new_geom

    def clean_tail(self, img, cam_geom):
        mask = tailcuts_clean(
                cam_geom, img,
                picture_thresh=self.tail_thresholds[cam_geom.cam_id][1],
                boundary_thresh=self.tail_thresholds[cam_geom.cam_id][0])
        if self.dilate:
            dilate(cam_geom, mask)
        img[~mask] = 0

        self.cutflow.count("tailcut cleaning")

        if "ASTRI" in cam_geom.cam_id:
            # turn into 2d to apply island cleaning
            img = astri_to_2d_array(img)

            # if set, remove all signal patches but the biggest one
            new_img = self.island_cleaning(img)

            if self.skip_edge_events:
                edge_thresh = np.max(new_img)/5.
                mask = np.ones_like(new_img, bool)
                mask[self.edge_width:-self.edge_width,
                     self.edge_width:-self.edge_width] = 0
                if (new_img[mask] > edge_thresh).any():
                    raise EdgeEven
                self.cutflow.count("tailcut edge")

            new_img = array_2d_to_astri(new_img)
            new_geom = cam_geom
        else:
            rot_geom, rot_img = convert_geometry_1d_to_2d(
                                    cam_geom, img, cam_geom.cam_id)

            cleaned_img = self.island_cleaning(rot_img, self.hex_neighbours_1ring)

            unrot_geom, unrot_img = convert_geometry_back(rot_geom, cleaned_img,
                                                          cam_geom.cam_id)
            new_img = unrot_img
            new_geom = unrot_geom

        return new_img, new_geom

    def clean_none(self, img, cam_geom):
        return img, cam_geom
