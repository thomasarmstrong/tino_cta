import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt


import irf_builder as irf


try:
    from matplotlib2tikz import save as tikzsave

    def tikz_save(arg, **kwargs):
        tikzsave(arg, **kwargs,
                 figureheight='\\figureheight',
                 figurewidth='\\figurewidth')
except ImportError:
    print("matplotlib2tikz is not installed")
    print("install with: \n$ pip install matplotlib2tikz")

    def tikz_save(arg, **kwargs):
        print("matplotlib2tikz is not installed")
        print("no .tex is saved")


# pull in all the plot functions into one `plotting` namespace
from irf_builder.sensitivity import plot_sensitivity
from irf_builder.irfs.effective_areas import plot_effective_areas
from irf_builder.irfs.angular_resolution import (plot_theta_square,
                                                 plot_angular_resolution,
                                                 plot_angular_resolution_violin)

# some dictionaries to control the visuals of the different plots in the same figure
channel_map = {'g': "gamma", 'p': "proton", 'e': "electron"}
channel_colour_map = {'g': "orange", 'p': "blue", 'e': "red"}
channel_marker_map = {'g': 's', 'p': '^', 'e': 'v'}

step_linestyle_map = {'theta': '-', 'gammaness': '--',
                      'reco': '-.', 'trigger': ':'}

mode_map = {"wave": "wavelets", "tail": "tailcuts"}
mode_colour_map = {"tail": "darkorange", "wave": "darkred"}


def save_fig(outname, endings=["tex", "pdf", "png"], **kwargs):
    """saves the current matplotlib figure as a given list of file formats

    Parameters
    ----------
    outname : string
        destination path without ending for the figure being saved
    endings : list of strings, optional (default: ["tex", "pdf", "png"])
        list of file name endings to save the figure as; needs to be eiter "tex" or
        any other ending supported and recognised by matplotlib

    Note
    ----
    uses the package `matplotlib2tikz` to save figures as tikz/pgf plots;
    if not installed, only prints an info message
    """
    for end in endings:
        if end == "tex":
            tikz_save(f"{outname}.{end}", **kwargs)
        else:
            plt.savefig(f"{outname}.{end}")


def plot_crab(e_bins=None, fractions=None):
    e_bins = e_bins or irf.e_bin_centres
    fractions = fractions or [1, 10, 100]
    for i, frac in enumerate(fractions):
        plt.loglog(e_bins, (irf.spectra.crab_source_rate(e_bins) / frac * e_bins**2)
                   .to(irf.sensitivity_unit).value,
                   color="red", ls="dashed", alpha=1 - (i / len(fractions)),
                   label="Crab Nebula" + (f" / {frac}" if frac != 1 else ""))


def plot_reference():
    """some pseude-official line to compare
    """
    ref_loge, ref_sens = *(np.array([
        (-1.8, 6.87978e-11), (-1.6, 1.87765e-11),
        (-1.4, 7.00645e-12), (-1.2, 1.77677e-12), (-1.0, 8.19263e-13),
        (-0.8, 4.84879e-13), (-0.6, 3.00256e-13), (-0.4, 2.07787e-13),
        (-0.2, 1.4176e-13), (0.0, 1.06069e-13), (0.2, 8.58209e-14),
        (0.4, 6.94294e-14), (0.6, 6.69301e-14), (0.8, 7.61169e-14),
        (1.0, 7.13895e-14), (1.2, 9.49376e-14), (1.4, 1.25208e-13),
        (1.6, 1.91209e-13), (1.8, 3.11611e-13), (2.0, 4.80354e-13)]).T),
    plt.loglog(10**ref_loge,
               ((ref_sens) * (u.erg * u.cm**2 * u.s)**(-1)).to(irf.flux_unit).value,
               marker="s", color="black", ms=3, linewidth=1,
               label="reference")
