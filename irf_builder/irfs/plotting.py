from matplotlib import pyplot as plt

import irf_builder as irf


channel_map = {'g': "gamma", 'p': "proton", 'e': "electron"}
channel_colour_map = {'g': "orange", 'p': "blue", 'e': "red"}
channel_marker_map = {'g': 's', 'p': '^', 'e': 'v'}
channel_linestyle_map = {'g': '-', 'p': '--', 'e': ':'}

mode_colour_map = {"tail": "darkorange", "wave": "darkred"}


def plot_lines(data, abscissa=None, labels=None, title=None, xlabel=None, ylabel=None,
               axis=None, xlog=True, ylog=True, grid=True):

    labels = labels or channel_map
    abscissa = abscissa or irf.e_bin_centres

    if axis:
        plt.sca(axis)
    else:
        plt.figure()
        axis = plt.gca()

    for cl, a in data.items():
        plt.plot(abscissa, a, label=labels[cl],
                 color=channel_colour_map[cl],
                 ls=channel_linestyle_map[cl],
                 marker=channel_marker_map[cl])
    plt.legend()

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlog:
        axis.set_xscale("log")
    if ylog:
        axis.set_yscale("log")
    if grid:
        plt.grid()


def plot_effective_areas(eff_areas, axis=None):
    if axis:
        plt.sca(axis)

    for cl, a in data.items():
        plt.plot(irf.e_bin_centres, a, label=channel_map[cl],
                 color=channel_colour_map[cl],
                 ls=channel_linestyle_map[cl],
                 marker=channel_marker_map[cl])
    plt.legend()
    plt.title(title)
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    plt.grid()
