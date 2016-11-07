from random import random
import numpy as np

from extract_and_crop_simtel_images import crop_astri_image

from ctapipe.image.cleaning import tailcuts_clean, dilate

''' old '''
from datapipe.denoising.wavelets_mrtransform import WaveletTransform as WaveletTransformOld
''' new '''
from datapipe.denoising.wavelets_mrfilter import WaveletTransform    as WaveletTransformNew


class UnknownModeException(Exception):
    pass


class EdgeEventException(Exception):
    pass


class ImageCleaner:

    def __init__(self, old=False, mode="wave", dilate=False,
                 skip_edge_events=True):
        self.old = old
        self.mode = mode
        self.dilate = dilate
        self.skip_edge_events = skip_edge_events
        self.wave_out_name = "/tmp/wavelet_{}_".format(random())
        self.wavelet_transform_old = WaveletTransformOld()
        self.wavelet_transform_new = WaveletTransformNew()

    def remove_isolated_pixels(self, img2d, threshold=0):
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

    def remove_plateau(self, img):
        img -= np.mean(img)
        img[img < 0] = 0

    def clean(self, img, tel_geom):

        if self.mode == "wave":
            # for now wavelet library works only on rectangular images
            cropped_img = crop_astri_image(img)
            if self.old:
                cleaned_img = \
                    self.wavelet_transform_old(cropped_img, 4,
                                               self.wave_out_name)
            else:
                cleaned_img = \
                    self.wavelet_transform_new(cropped_img)

            ''' wavelet_transform still leaves some isolated pixels;
                remove '''
            self.remove_isolated_pixels(cleaned_img)

            if self.old:
                ''' old wavelet_transform did leave constant background;
                    remove '''
                self.remove_plateau(cleaned_img)

            if self.skip_edge_events:
                edge_thresh = np.max(cleaned_img)/5.
                if (cleaned_img[0,:]  > edge_thresh).any() or  \
                   (cleaned_img[-1,:] > edge_thresh).any() or  \
                   (cleaned_img[:,0]  > edge_thresh).any() or  \
                   (cleaned_img[:,-1] > edge_thresh).any():
                        raise EdgeEventException

            img = cleaned_img.flatten()
            ''' hillas parameter function requires image and x/y arrays
                to be of the same dimension '''
            pix_x = crop_astri_image(tel_geom.pix_x).flatten()
            pix_y = crop_astri_image(tel_geom.pix_y).flatten()

        elif self.mode == "tail":
            mask = tailcuts_clean(tel_geom, img, 1,
                                  picture_thresh=10.,
                                  boundary_thresh=5.)
            if self.dilate:
                dilate(tel_geom, mask)
            img[mask == False] = 0

            '''
            events with too much signal at the edge might negatively
            influence hillas parametrisation '''
            if self.skip_edge_events:
                skip_event = False
                for pixid in tel_geom.pix_id[mask]:
                    if len(tel_geom.neighbors) < 8:
                        skip_event = True
                        break
                if skip_event:
                    raise EdgeEventException

            '''
            since wavelet transform crops pixel lists and returns them
            rename them here too for easy return '''
            pix_x, pix_y = tel_geom.pix_x, tel_geom.pix_y

        elif self.mode == "none":
            pix_x, pix_y = tel_geom.pix_x, tel_geom.pix_y
        else:
            raise UnknownModeException(
                'cleaning mode "{}" not found'.format(self.mode))

        return img, pix_x, pix_y
