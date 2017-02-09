import numpy as np

from copy import copy

from scipy import ndimage
from matplotlib import pyplot as plt

from ctapipe.image.cleaning import tailcuts_clean, dilate

from .CutFlow import CutFlow

try:
    from ctapipe.image.geometry_converter import convert_geometry_1d_to_2d, \
                                                 convert_geometry_back
except:
    print("Wrong version of ctapipe ; cannot handle hexagonal cameras.")


from sys import path
from os.path import expandvars
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/sap-cta-data-pipeline/"))
path.append(expandvars("$CTA_SOFT/"
            "jeremie_cta/snippets/ctapipe/"))
from datapipe.denoising.wavelets_mrfilter import WaveletTransform
from extract_and_crop_simtel_images import crop_astri_image


class UnknownModeException(Exception):
    pass


class EdgeEventException(Exception):
    pass


class MissingImplementationException(Exception):
    pass


def kill_isolpix(array, neighbours=None, threshold=.2, plot=False):
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

    if plot:
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))

        ax[0].imshow(np.sqrt(array), interpolation='none')
        ax[0].set_title('input image')

        ax[1].imshow(label_im, interpolation='none')
        ax[1].set_title('connected regions labels')

        ax[2].imshow(np.sqrt(filtered_array), interpolation='none')
        ax[2].set_title('cleaned output')

        for i, a in enumerate(ax): a.set_axis_off()
        plt.show()

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
                 skip_edge_events=True, cutflow=CutFlow("ImageCleaner"),
                 wavelet_options="-K -C1 -m3 -s3 -n4",
                 tail_thresh_up=10, tail_thresh_low=5):
        self.mode = mode
        self.skip_edge_events = skip_edge_events
        self.cutflow = cutflow

        if mode is None:
            self.clean = self.clean_none
        elif mode == "wave":
            self.clean = self.clean_wave
            self.wavelet_transform = WaveletTransform()
            self.wavelet_options = wavelet_options
            self.island_threshold = 2

            self.wavelet_options = "-K -C1 -m3 -s3 -n4"
        elif mode == "tail":
            self.clean = self.clean_tail
            self.tail_thresh_up = tail_thresh_up
            self.tail_thresh_low = tail_thresh_low
            self.island_threshold = 3.5
            self.dilate = dilate
        else:
            raise UnknownModeException(
                'cleaning mode "{}" not found'.format(mode))

        if island_cleaning:
            self.island_cleaning = kill_isolpix
        else:
            self.island_cleaning = lambda x, *args, **kw: x

    def clean_wave(self, img, cam_geom, foclen):
        if cam_geom.cam_id == "ASTRI":
            cropped_img = crop_astri_image(img)
            cleaned_img = self.wavelet_transform.clean_image(
                            cropped_img, raw_option_string=self.wavelet_options)

            self.cutflow.count("wavelet cleaning")

            ''' wavelet_transform still leaves some isolated pixels; remove them '''
            cleaned_img = self.island_cleaning(cleaned_img)

            if self.skip_edge_events:
                edge_thresh = np.max(cleaned_img)/5.
                if (cleaned_img[0,:]  > edge_thresh).any() or  \
                   (cleaned_img[-1,:] > edge_thresh).any() or  \
                   (cleaned_img[:,0]  > edge_thresh).any() or  \
                   (cleaned_img[:,-1] > edge_thresh).any():
                        raise EdgeEventException
                self.cutflow.count("wavelet edge")

            new_img = cleaned_img.flatten()
            new_geom = copy(cam_geom)
            new_geom.pix_x = crop_astri_image(cam_geom.pix_x).flatten()
            new_geom.pix_y = crop_astri_image(cam_geom.pix_y).flatten()
            new_geom.pix_area = np.ones_like(new_img) * cam_geom.pix_area[0]

        elif cam_geom.pix_type.startswith("hex"):
            try:
                rot_geom, rot_img = convert_geometry_1d_to_2d(
                                        cam_geom, img, cam_geom.cam_id)

                cleaned_img = self.wavelet_transform(
                        rot_img,
                        raw_option_string=self.wavelet_options)

                self.cutflow.count("wavelet cleaning")

                cleaned_img = self.island_cleaning(cleaned_img,
                                                   neighbours=self.hex_neighbours_1ring,
                                                   threshold=self.island_threshold)

                unrot_geom, unrot_img = convert_geometry_back(
                                        rot_geom, cleaned_img, cam_geom.cam_id, foclen)

                new_img = unrot_img
                new_geom = unrot_geom
            except:
                print("Wrong version of ctapipe ; cannot handle hexagonal cameras.")
        else:
            raise MissingImplementationException("wavelet cleaning of square-pixel"
                                                 " images only for ASTRI so far")

        return new_img, new_geom

    def clean_tail(self, img, cam_geom, foclen):
        mask = tailcuts_clean(cam_geom, img, 1,
                              picture_thresh=self.tail_thresh_up,
                              boundary_thresh=self.tail_thresh_low)
        if self.dilate:
            dilate(cam_geom, mask)
        img[mask == False] = 0

        self.cutflow.count("tailcut cleaning")

        if cam_geom.cam_id == "ASTRI":
            img = crop_astri_image(img)

            # if set, remove all signal patches but the biggest one
            new_img = self.island_cleaning(img)

            if self.skip_edge_events:
                edge_thresh = np.max(new_img)/5.
                if (new_img[0,:]  > edge_thresh).any() or  \
                   (new_img[-1,:] > edge_thresh).any() or  \
                   (new_img[:,0]  > edge_thresh).any() or  \
                   (new_img[:,-1] > edge_thresh).any():
                        raise EdgeEventException
                self.cutflow.count("tailcut edge")

            new_img = new_img.flatten()
            new_geom = copy(cam_geom)
            new_geom.pix_x = crop_astri_image(cam_geom.pix_x).flatten()
            new_geom.pix_y = crop_astri_image(cam_geom.pix_y).flatten()
            new_geom.pix_area = np.ones_like(new_img) * cam_geom.pix_area[0]
        else:
            rot_geom, rot_img = convert_geometry_1d_to_2d(
                                    cam_geom, img, cam_geom.cam_id)

            cleaned_img = self.island_cleaning(rot_img, self.hex_neighbours_1ring)

            unrot_geom, unrot_img = convert_geometry_back(
                                    rot_geom, cleaned_img, cam_geom.cam_id, foclen)
            new_img = unrot_img
            new_geom = unrot_geom

        return new_img, new_geom

    def clean_none(self, img, cam_geom, *args):
        return img, cam_geom
