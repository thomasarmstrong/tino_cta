from random import random
import numpy as np

from scipy import ndimage
from matplotlib import pyplot as plt

from extract_and_crop_simtel_images import crop_astri_image

from ctapipe.image.cleaning import tailcuts_clean, dilate

from datapipe.denoising.wavelets_mrfilter import WaveletTransform

from .CutFlow import CutFlow

from ctapipe.io import convert_geometry_1d_to_2d, convert_geometry_back

class UnknownModeException(Exception):
    pass


class EdgeEventException(Exception):
    pass


def kill_isolpix(array, plot=False):
    """ Return array with isolated islands removed.
        Only keeping the biggest islands (largest surface).
    :param array: Array with completely isolated cells
    :param struct: Structure array for generating unique regions
    :return: Filtered array with just the largest island
    """

    filtered_array = np.copy(array)

    filtered_array[filtered_array < 0.2] = 0
    mask = filtered_array > 0

    label_im, nb_labels = ndimage.label(mask)

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


def remove_isolated_pixels(img2d, threshold=0):
    max_val = np.max(img2d)
    for idx, foo in enumerate(img2d):
        for idy, bar in enumerate(foo):
            threshold = 3
            is_island = 0
            if idx > 0:
                is_island += img2d[idx-1, idy]
            if idx < len(img2d)-1:
                is_island += img2d[idx+1, idy]
            if idy > 0:
                is_island += img2d[idx, idy-1]
            if idy < len(foo)-1:
                is_island += img2d[idx, idy+1]

            if is_island < threshold and bar != max_val:
                img2d[idx, idy] = 0


class ImageCleaner:

    def __init__(self, old=False, mode="wave", dilate=False,
                 skip_edge_events=True, cutflow=CutFlow("ImageCleaner")):
        self.old = old
        self.mode = mode
        self.dilate = dilate
        self.skip_edge_events = skip_edge_events
        self.wave_out_name = "/tmp/wavelet_{}_".format(random())
        self.wavelet_transform = WaveletTransform()
        self.cutflow = cutflow

        if self.mode == "wave":
            self.clean = self.clean_wave
        elif self.mode == "tail":
            self.clean = self.clean_tail
        elif self.mode == "none":
            self.clean = self.clean_none
        else:
            raise UnknownModeException(
                'cleaning mode "{}" not found'.format(self.mode))

    def remove_plateau(self, img):
        img -= np.mean(img)
        img[img < 0] = 0

    def clean_wave(self, img, cam_geom, foclen):
        if cam_geom.cam_id == "ASTRI":
            cropped_img = crop_astri_image(img)
            cleaned_img = self.wavelet_transform(cropped_img)

            self.cutflow.count("wavelet cleaning")

            ''' wavelet_transform still leaves some isolated pixels; remove them '''
            cleaned_img = kill_isolpix(cleaned_img)

            if self.old:
                ''' old wavelet_transform did leave constant background; remove '''
                self.remove_plateau(cleaned_img)

            if self.skip_edge_events:
                edge_thresh = np.max(cleaned_img)/5.
                if (cleaned_img[0,:]  > edge_thresh).any() or  \
                   (cleaned_img[-1,:] > edge_thresh).any() or  \
                   (cleaned_img[:,0]  > edge_thresh).any() or  \
                   (cleaned_img[:,-1] > edge_thresh).any():
                        raise EdgeEventException

            self.cutflow.count("reject edge events")

            img = cleaned_img.flatten()
            ''' hillas parameter function requires image and x/y arrays
                to be of the same dimension '''
            pix_x = crop_astri_image(cam_geom.pix_x).flatten()
            pix_y = crop_astri_image(cam_geom.pix_y).flatten()

        elif cam_geom.pix_type.startswith("hex"):
            rot_geom, rot_img = convert_geometry_1d_to_2d(
                                    cam_geom, img, cam_geom.cam_id)

            cleaned_img = self.wavelet_transform(rot_img)

            self.cutflow.count("wavelet cleaning")

            cleaned_img = kill_isolpix(cleaned_img)

            unrot_geom, unrot_img = convert_geometry_back(
                                    rot_geom, cleaned_img, cam_geom.cam_id, foclen)

            img = unrot_img.flatten()
            pix_x = unrot_geom.pix_x
            pix_y = unrot_geom.pix_y

        return img, pix_x, pix_y

    def clean_tail(self, img, cam_geom, foclen):
        mask = tailcuts_clean(cam_geom, img, 1,
                              picture_thresh=10.,
                              boundary_thresh=5.)
        if self.dilate:
            dilate(cam_geom, mask)
        img[mask == False] = 0

        if cam_geom.cam_id == "ASTRI":
            cleaned_img = crop_astri_image(img)

            if self.skip_edge_events:
                edge_thresh = np.max(cleaned_img)/5.
                if (cleaned_img[0,:]  > edge_thresh).any() or  \
                   (cleaned_img[-1,:] > edge_thresh).any() or  \
                   (cleaned_img[:,0]  > edge_thresh).any() or  \
                   (cleaned_img[:,-1] > edge_thresh).any():
                        raise EdgeEventException

            img = cleaned_img.flatten()
            pix_x = crop_astri_image(cam_geom.pix_x).flatten()
            pix_y = crop_astri_image(cam_geom.pix_y).flatten()
        else:
            pix_x = cam_geom.pix_x
            pix_y = cam_geom.pix_y

        #'''
        #events with too much signal at the edge might negatively
        #influence hillas parametrisation '''
        #if self.skip_edge_events:
            #skip_event = False
            #for pixid in tel_geom.pix_id[mask]:
                #if len(tel_geom.neighbors) < 8:
                    #skip_event = True
                    #break
            #if skip_event:
                #raise EdgeEventException
        #'''
        #since wavelet transform crops pixel lists and returns them
        #rename them here too for easy return '''
        #pix_x, pix_y = tel_geom.pix_x, tel_geom.pix_y

        self.cutflow.count("reject edge events")

        return img, pix_x, pix_y

    def clean_none(self, img, cam_geom, foclen):
        return img, cam_geom.pix_x, cam_geom.pix_y
