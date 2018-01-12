import numpy as np
from matplotlib import pyplot as plt

import irf_builder as irf


def get_energy_migration_matrix(events):
    """
    Return
    ------
    energy_matrix : 2D array
        the energy migration matrix in the form of:
        `energy_matrix[mc_energy_bin][reco_energy_bin]`
    """
    energy_matrix = {}
    for i, (channel, evt) in enumerate(events.items()):
        counts, _, _ = np.histogram2d(evt[irf.mc_energy_name],
                                      evt[irf.reco_energy_name],
                                      bins=(irf.e_bin_edges_fine,
                                            irf.e_bin_edges_fine))
        energy_matrix[channel] = counts
    return energy_matrix


def plot_energy_migration_matrix(energy_matrix, fig=None):
    if fig is None:
        fig = plt.gcf()
    for i, (channel, e_matrix) in enumerate(energy_matrix.items()):
        ax = fig.add_subplot(131 + i)

        ax.pcolormesh(irf.e_bin_edges_fine.value,
                      irf.e_bin_edges_fine.value, e_matrix)
        plt.plot(irf.e_bin_edges_fine.value[[0, -1]],
                 irf.e_bin_edges_fine.value[[0, -1]],
                 color="darkgreen")
        plt.title(irf.plotting.channel_map[channel])
        ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
        if i == 0:
            ax.set_ylabel(r"$E_\mathrm{MC}$ / TeV")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.grid()

    plt.tight_layout()
    # plt.subplots_adjust(top=0.921, bottom=0.148,
    #                     left=0.093, right=0.982,
    #                     hspace=0.2, wspace=0.27)


def get_rel_delta_e(events, ref_energy_name=None):
    ref_energy_name = ref_energy_name or irf.reco_energy_name
    rel_delta_e = {}
    for ch in events:
        counts, _, _ = np.histogram2d(
            events[ch][ref_energy_name],
            (events[ch][irf.reco_energy_name] - events[ch][irf.mc_energy_name]) /
            events[ch][ref_energy_name],
            bins=(irf.e_bin_edges_fine, np.linspace(-1, 1, 100)))
        rel_delta_e[ch] = counts
    return rel_delta_e


def plot_rel_delta_e(rel_delta_e, fig=None):
    if fig is None:
        fig = plt.gcf()
    for i, ch in enumerate(rel_delta_e):
        ax = fig.add_subplot(131 + i)
        ax.pcolormesh(irf.e_bin_edges_fine / irf.energy_unit,
                      np.linspace(-1, 1, 100),
                      rel_delta_e[ch].T)
        plt.plot(irf.e_bin_edges_fine.value[[0, -1]], [0, 0],
                 color="darkgreen")
        plt.title(irf.plotting.channel_map[ch])
        ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
        if i == 0:
            ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{reco}$")
        ax.set_xscale("log")
        plt.grid()

    plt.tight_layout()
    # plt.subplots_adjust(top=0.921, bottom=0.148,
    #                     left=0.105, right=0.982,
    #                     hspace=0.2, wspace=0.332)


def get_energy_bias(events):
    energy_bias = {}
    for ch, e in events.items():
        median_bias = np.zeros_like(irf.e_bin_centres.value)
        for i, (e_low, e_high) in enumerate(zip(irf.e_bin_edges[:-1] / irf.energy_unit,
                                                irf.e_bin_edges[1:] / irf.energy_unit)):
            bias = (e[irf.mc_energy_name] / e[irf.reco_energy_name]) - 1

            try:
                median_bias[i] = np.percentile(
                    bias[(e[irf.reco_energy_name] > e_low) &
                         (e[irf.reco_energy_name] < e_high)],
                    50)
            except IndexError:
                pass
        energy_bias[ch] = median_bias

    return energy_bias


def plot_energy_bias(energy_bias, channels=None):
    channels = channels or ['g']
    for i, ch in enumerate(channels):
        plt.gcf().add_subplot(111 + i)
        plt.plot(irf.e_bin_centres, energy_bias[ch],
                 color=irf.plotting.channel_colour_map[ch],
                 label=irf.plotting.channel_map[ch])
        plt.gca().set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.gcf().axes[0].set_ylabel(r"$E_\mathrm{MC}/E_\mathrm{reco} - 1$")
    plt.xscale('log')
    plt.grid()


def correct_energy_bias(events, energy_bias):
    from scipy import interpolate

    spline = interpolate.splrep(irf.e_bin_centres.value, energy_bias, k=1)

    for ch in events:
        events[ch][irf.reco_energy_name] = \
            events[ch][irf.reco_energy_name] * \
            (1 + interpolate.splev(events[ch][irf.reco_energy_name], spline))
    return events


def get_energy_resolution(events, ref_energy_name=None, percentile=68):
    ref_energy_name = ref_energy_name or irf.reco_energy_name
    energy_resolution = {}
    for ch, e in events.items():
        resolution = np.zeros_like(irf.e_bin_centres.value)
        for i, (e_low, e_high) in enumerate(zip(irf.e_bin_edges[:-1] / irf.energy_unit,
                                                irf.e_bin_edges[1:] / irf.energy_unit)):
            rel_error = np.abs(e[irf.mc_energy_name] - e[irf.reco_energy_name]) / \
                e[ref_energy_name]

            try:
                resolution[i] = np.percentile(
                    rel_error[(e[ref_energy_name] > e_low) &
                              (e[ref_energy_name] < e_high)],
                    percentile)
            except IndexError:
                pass
        energy_resolution[ch] = resolution

    return energy_resolution


def plot_energy_resolution(energy_resolution):
    # for ch in energy_resolution:
    for ch in ['g']:
        plt.plot(irf.e_bin_centres, energy_resolution[ch],
                 label=irf.plotting.channel_map[ch],
                 color=irf.plotting.channel_colour_map[ch])
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$")
    plt.gca().set_xscale("log")
    plt.legend()
    plt.grid()
