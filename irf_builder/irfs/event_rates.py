import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import irf_builder as irf


def get_simulated_energy_distribution(generator_spectra, n_simulated_events):
    generator_energy_hists = {}
    for cl in generator_spectra:
        generator_energy_hists[cl] = irf.spectra.make_mock_event_rate(
            generator_spectra[cl], norm=n_simulated_events[cl],
            bin_edges=irf.e_bin_edges, log_e=False)
    return generator_energy_hists


def get_simulated_energy_distribution_wrapper(events):
    """
    Notes
    -----
    solely depends on meta data -- `events` is not actually used -- only here to unify
    interface for the function calls
    """
    return get_simulated_energy_distribution(
        generator_spectra={'g': irf.spectra.e_minus_2,
                           'p': irf.spectra.e_minus_2,
                           'e': irf.spectra.e_minus_2},
        n_simulated_events={'g': irf.meta_data["gamma"]["n_simulated"],
                            'p': irf.meta_data["proton"]["n_simulated"],
                            'e': irf.meta_data["electron"]["n_simulated"]}
    )


def plot_energy_distribution(events=None, energies=None,
                             e_bin_edges=None):
    if (events is None) == (energies is None):
        raise ValueError("please provide one of `events` or `energies`, but not both")

    if energies is None:
        energies = dict((c, e[irf.mc_energy_name]) for c, e in events.items())

    e_bin_edges = e_bin_edges or irf.e_bin_edges

    for ch, energy in energies.items():
        plt.bar(e_bin_edges[:-1], energy, width=np.diff(e_bin_edges), align='edge',
                label=irf.plotting.channel_map[ch],
                color=irf.plotting.channel_colour_map[ch],
                alpha=.5)
    plt.legend()
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel("number of generated events")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()


def get_energy_event_rates(events, th_cuts, e_bin_edges=None, energy_name=None):
    energy_name = energy_name or irf.reco_energy_name
    e_bin_edges = e_bin_edges or irf.e_bin_edges
    energy_rates = {}
    for ch, e in events.items():
        counts = np.histogram(e[energy_name], bins=e_bin_edges,
                              weights=e["weight"])[0]

        angle = th_cuts * (1 if ch == 'g' else irf.r_scale) * u.deg
        angular_area = 2 * np.pi * (1 - np.cos(angle)) * u.sr
        energy_rates[ch] = counts / (angular_area.to(u.deg**2) *
                                     irf.observation_time.to(u.s))
    return energy_rates


def plot_energy_event_rates(energy_rates, e_bin_edges=None):
    e_bin_edges = e_bin_edges or irf.e_bin_edges
    for ch in energy_rates:
        plt.plot(e_bin_edges[:-1], energy_rates[ch],
                 label=irf.plotting.channel_map[ch],
                 color=irf.plotting.channel_colour_map[ch],
                 )
    plt.legend()
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"event rate $f / (\mathrm{s}^{-1}*\mathrm{deg}^{-2})$")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.tight_layout()
