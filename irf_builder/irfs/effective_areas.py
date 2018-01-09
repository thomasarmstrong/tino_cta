import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import irf_builder as irf
from irf_builder.spectra import make_mock_event_rate, e_minus_2


def get_effective_areas(events, generator_areas,
                        n_simulated_events=None,
                        generator_spectra=None,
                        generator_energy_hists=None
                        ):
    """
    calculates the effective areas for the provided channels

    Parameters
    ----------
    generator_areas : astropy quantities
        the area for each channel within which the shower impact position was
        generated
    n_simulated_events : dictionary of integers, optional (defaults: None)
        number of events used in the MC simulation for each channel
    generator_spectra : dictionary of functors, optional (default: None)
        function object for the differential generator flux of each channel
    generator_energy_hists : numpy arrays, optional (default: None)
        energy histogram of the generated events for each channel binned according to
        `.energy_bin_edges`

    Returns
    -------
    eff_area_gam, eff_area_pro : numpy arrays
        histograms of the effective areas of gammas and protons binned according to
        `.bin_edges_gam` and `.bin_edges_pro`

    Notes
    -----
    either give the histogram of the energy distributions at MC generator level with
    `generator_energy_hists` or create them on the fly with `n_simulated_events` and
    `spectra`
    """

    if (n_simulated_events is not None and generator_spectra is not None) == \
            (generator_energy_hists):
        raise ValueError("use either (n_simulated_events and generator"
                         "_spectra) or generator_energy_hists to set the MC "
                         "generated energy spectrum -- not both")

    if not generator_energy_hists:
        generator_energy_hists = {}
        # generate the histograms for the energy distributions of the Monte Carlo
        # generated events given the generator spectra and the number of generated
        # events
        for cl in events:
            generator_energy_hists[cl] = make_mock_event_rate(
                generator_spectra[cl], norm=n_simulated_events[cl],
                bin_edges=irf.e_bin_edges, log_e=False)

    # an energy-binned histogram of the effective areas
    # binning according to .energy_bin_edges[cl]
    effective_areas = {}

    # an energy-binned histogram of the selected events
    # binning according to .energy_bin_edges[cl]
    selected_events = {}

    # generate the histograms for the energy distributions of the selected
    # events
    for cl in events:
        mc_energy = events[cl][irf.mc_energy_name].values * irf.energy_unit
        selected_events[cl] = np.histogram(mc_energy, bins=irf.e_bin_edges)[0]

        # the effective areas are the selection efficiencies per energy bin multiplied
        # by the area in which the Monte Carlo events have been generated
        # in
        efficiency = selected_events[cl] / generator_energy_hists[cl]
        effective_areas[cl] = efficiency * generator_areas[cl]

    return effective_areas


def get_effective_areas_wrapper(events):
    return get_effective_areas(
        events,
        generator_areas={'g': np.pi * (irf.meta_data["gamma"]["gen_radius"] * u.m)**2,
                         'p': np.pi * (irf.meta_data["proton"]["gen_radius"] * u.m)**2,
                         'e': np.pi * (irf.meta_data["electron"]["gen_radius"] * u.m)**2},
        n_simulated_events={'g': irf.meta_data["gamma"]["n_simulated"],
                            'p': irf.meta_data["proton"]["n_simulated"],
                            'e': irf.meta_data["electron"]["n_simulated"]},
        generator_spectra={'g': e_minus_2,
                           'p': e_minus_2,
                           'e': e_minus_2}
    )


def plot_effective_areas(eff_areas):
    """plots the effective areas of the different channels as a line plot

    Parameter
    ---------
    eff_areas : dict of 1D arrays
        dictionary of the effective areas of the different channels
    """
    for cl, a in eff_areas.items():
        plt.plot(irf.e_bin_centres, a,
                 label=irf.plotting.channel_map[cl],
                 color=irf.plotting.channel_colour_map[cl],
                 marker=irf.plotting.channel_marker_map[cl])
    plt.legend()
    plt.title("Effective Areas")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()
