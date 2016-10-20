import numpy as np


import signal
class SignalHandler():
    def __init__(self):
        self.stop = False

    def __call__(self, signal, frame):
        if self.stop:
            print('you pressed Ctrl+C again -- exiting NOW')
            exit(-1)
        print('you pressed Ctrl+C!')
        print('exiting after current event')
        self.stop = True


import pyhessio
def apply_mc_calibration_ASTRI(adcs, tel_id, mode=0, adc_tresh=3500):
    """
    apply basic calibration for ASTRI telescopes with two gains
    """
    gains0 = pyhessio.get_calibration(tel_id)[0]
    gains1 = pyhessio.get_calibration(tel_id)[1]

    if mode == 0:
        ''' old mode -- pedestal array was broken, had to be put in by hand '''
        calibrated = [(adc0 - 971)*gain0 if adc0 < adc_tresh
                      else (adc1 - 961)*gain1 for adc0, adc1, gain0, gain1
                      in zip(adcs[0], adcs[1], gains0, gains1)]
    else:
        ''' new mode -- pedestal values should be fixed now'''
        peds0 = pyhessio.get_pedestal(tel_id)[0]
        peds1 = pyhessio.get_pedestal(tel_id)[1]
        calibrated = [(adc0-ped0)*gain0 if adc0 < adc_tresh
                      else (adc1-ped1)*gain1
                      for adc0, adc1, gain0, gain1, ped0, ped1
                      in zip(adcs[0], adcs[1], gains0, gains1, peds0, peds1)]

    return np.array(calibrated)


from astropy import units as u
def convert_astropy_array(arr, unit=None):
    if unit is None: unit = arr[0].unit
    return np.array([a.to(unit).value for a in arr])*unit


import argparse
def make_argparser():
    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-c', '--min_charge', type=int, default=0,
                        help="minimum charge per telescope after cleaning")
    parser.add_argument('-i', '--indir',   type=str,
                        default="/local/home/tmichael/Data/cta/ASTRI9/gamma/")
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave",
                        help="if set, use tail cleaning, otherwise wavelets")
    parser.add_argument('--dilate', default=False, action='store_true',
                        help="use dilate function for tailcut cleaning")
    parser.add_argument('-t', '--teltype', type=str, default="all")
    parser.add_argument('-o', '--outtoken', type=str, default=None,
                        help="helper token; useful for refining filenames")
    parser.add_argument('-w', '--write', action='store_true',
                        help="write output -- e.g. plots, classifiers, events")
    parser.add_argument('-p', '--plot',  action='store_true',
                        help="do some plotting")
    parser.add_argument('-d', '--dry', dest='last', action='store_const',
                        const=1, default=-1,
                        help="only consider first file per type")
    return parser



from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from ctapipe import visualization
continue_drawing = True
#func_figure = plt.figure()
def draw_image(tel_geom, pmt_signal, moments=None, pix_x=None, pix_y=None):
    global continue_drawing
    global func_figure
    if continue_drawing:
        ax = plt.subplot(111)
        try:
            disp = visualization.CameraDisplay(tel_geom, ax=ax)
            disp.image = pmt_signal
            disp.cmap = plt.cm.hot
            disp.add_colorbar()
            if moments:
                disp.overlay_moments(moments, color='seagreen', linewidth=3)
        except ValueError:
            plt.imshow(pmt_signal.reshape(40, 40), interpolation='none',
                       #extent=(min(pix_x).value, max(pix_x).value,
                               #min(pix_y).value, max(pix_y).value)
                       )

            #ellipse = Ellipse(xy=(moments.cen_x, moments.cen_y),
                              #width=moments.width, height=moments.length,
                              #angle=np.degrees(moments.phi), fill=False)
            #ax.add_patch(ellipse)

        plt.pause(.1)

        print("[enter] for next event")
        print("anyting else: break rawing")
        response = input("Choice: ")
        if response != "":
            continue_drawing = False
