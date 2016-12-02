from os.path import expandvars

import numpy as np

from astropy import units as u

import signal
class SignalHandler():
    ''' handles ctrl+c signals; set up via
        signal_handler = SignalHandler()
        signal.signal(signal.SIGINT, signal_handler)
    '''
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
def apply_mc_calibration_ASTRI(adcs, gains, peds, mode=0, adc_tresh=3500):
    """
    apply basic calibration for ASTRI telescopes with two gains
    """
    gains0 = gains[0]
    gains1 = gains[1]

    peds0 = peds[0]
    peds1 = peds[1]

    calibrated = [(adc0-ped0)*gain0 if adc0 < adc_tresh
                    else (adc1-ped1)*gain1
                    for adc0, adc1, gain0, gain1, ped0, ped1
                    in zip(adcs[0], adcs[1], gains0, gains1, peds0, peds1)]

    return np.array(calibrated)


def apply_mc_calibration(adcs, gains, peds):
    """
    apply basic calibration
    """

    if adcs.ndim > 1:  # if it's per-sample need to correct the peds
        return ((adcs - peds[:, np.newaxis] / adcs.shape[1]) *
                gains[:, np.newaxis])

    return (adcs - peds) * gains


def convert_astropy_array(arr, unit=None):
    if unit is None: unit = arr[0].unit
    return (np.array([a.to(unit).value for a in arr])*unit).si


import argparse
def make_argparser():
    parser = argparse.ArgumentParser(description='show single telescope')
    parser.add_argument('-m', '--max_events', type=int, default=None,
                        help="maximum number of events considered per file")
    parser.add_argument('-c', '--min_charge', type=int, default=0,
                        help="minimum charge per telescope after cleaning")
    parser.add_argument('-i', '--indir',   type=str,
                        default=expandvars("$HOME/Data/cta/ASTRI9/"))
    parser.add_argument('-r', '--runnr',   type=str, default="*")
    parser.add_argument('--tail', dest="mode", action='store_const',
                        const="tail", default="wave",
                        help="if set, use tail cleaning, otherwise wavelets")
    parser.add_argument('--dilate', default=False, action='store_true',
                        help="use dilate function for tailcut cleaning")
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
        print("anyting else: break drawing")
        response = input("Choice: ")
        if response != "":
            continue_drawing = False


from matplotlib2tikz import save as tikzsave
def tikz_save(arg, **kwargs):
    tikzsave(arg, figureheight = '\\figureheight',
                  figurewidth  = '\\figurewidth', **kwargs)
def save_fig(outname, endings=["tex", "pdf", "png"], **kwargs):
    for end in endings:
        if end == "tex":
            tikz_save("{}.{}".format(outname, end), **kwargs)
        else:
            ptl.savefig("{}.{}".format(outname, end))


def make_mock_event_rate(spectra, binEdges=None, Emin=None, Emax=None,
                         E_unit=u.GeV, NBins=None, logE=True, norm=None):

    rates = [[] for f in spectra]

    if binEdges is None:
        if logE:
            Emin = np.log10(Emin/E_unit)
            Emax = np.log10(Emax/E_unit)
        binEdges = np.linspace(Emin, Emax, NBins+1, True)

    for l_edge, h_edge in zip(binEdges[:-1], binEdges[1:]):
        if logE:
            bin_centre = 10**((l_edge+h_edge)/2.) * E_unit
            bin_width = (10**h_edge-10**l_edge)*E_unit

        else:
            bin_centre = (l_edge+h_edge) * E_unit / 2.
            bin_width = (h_edge-l_edge)*E_unit
        for i, spectrum in enumerate(spectra):
            bin_events = spectrum(bin_centre) * bin_width
            rates[i].append(bin_events)

    for i, rate in enumerate(rates):
        rate = convert_astropy_array(rate)
        if norm:
            rate *= norm[i]/np.sum(rate)
        rates[i] = rate

    return (*rates), binEdges

# ================================== #
# Compute Eq. (17) of Li & Ma (1983) #
# ================================== #
def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute Eq. (17) of Li & Ma (1983).

    Parameters:
     Non   - Number of on counts
     Noff  - Number of off counts
    Keywords:
     alpha - Ratio of on-to-off exposure
    """

    alpha1 = alpha + 1.0
    sum    = Non + Noff
    arg1   = Non / sum
    arg2   = Noff / sum
    term1  = Non  * np.log((alpha1/alpha)*arg1)
    term2  = Noff * np.log(alpha1*arg2)
    sigma  = np.sqrt(2.0 * (term1 + term2))

    return sigma




def plot_hex_and_violine(abscissa, ordinate, bin_edges, extent=None, vmin=None, vmax=None,
                         xlabel="", ylabel="", do_hex=True, do_violine=True,
                         cm=plt.cm.hot):

    val_vs_dep = {}
    bin_centres = (bin_edges[1:]+bin_edges[:-1])/2.
    for dep, val in zip(abscissa, ordinate):
        ''' get the bin number this event belongs into '''
        ibin = np.digitize(dep, bin_edges)-1
        ibin = min(ibin, len(bin_centres)-1)

        ''' the central value of the bin is the key for the dictionary '''
        if bin_centres[ibin] not in val_vs_dep:
            val_vs_dep[bin_centres[ibin]]  = [val]
        else:
            val_vs_dep[bin_centres[ibin]] += [val]

    plt.figure()
    if do_hex and do_violine:
        plt.subplot(211)
    if do_hex:
        plt.hexbin(abscissa,
                ordinate,
                vmax=vmax,
                gridsize=40,
                extent=extent,
                cmap=cm)
        plt.colorbar()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if do_hex and do_violine:
        plt.subplot(212)

    if do_violine:
        vals = [a for a in val_vs_dep.values()]
        keys = [a for a in val_vs_dep.keys()]
        try:
            widths=bin_edges[1]-bin_edges[0]
        except IndexError:
            widths = 1

        plt.violinplot(vals, keys,
                    points=60, widths=widths,
                    showextrema=True, showmedians=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()






if __name__ == "__main__":

    def Eminus2(e, unit=u.GeV):
        return (e/unit)**(-2) / (unit * u.s * u.m**2)

    import sys
    ebins = np.linspace(2,8,7,True)
    rate, binEdges = make_mock_event_rate([Eminus2], logE=int(sys.argv[1]),
                                          Emin=1e2, Emax=1e8, NBins=16,
                                          #binEdges=ebins,
                                          norm=1)
    print(binEdges)
    figure = plt.figure()
    #plt.plot(marker='o',
    plt.bar(
        (binEdges[1:]+binEdges[:-1])/2, rate[0].value )
    plt.yscale('log')
    if not int(sys.argv[1]):
        plt.xscale('log')
    plt.show()

