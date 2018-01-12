import numpy as np
from astropy import units as u

import irf_builder as irf


def unbinned(events, n_simulated_events, e_min_max, spectra,
             generator_areas, generator_gamma, observation_time,
             diff_angle=None, extensions=None):
    """
    generates a weight for every event

    Parameters
    ----------
    n_simulated_events : dictionary
        total number of simulated events for every channel
    e_min_max : dictionary of tuples
        lower and upper energy limits used in the Monte Carlo event generator
    generator_areas : dictionary
        area within which the shower impact point has been distributed by the
        Monte Carlo event generator
    generator_gamma : dictionary
        index of the MC generated energy spectrum
    observation_time : astropy time quantity
        set the exposure time the weighted events should correspond to
    diff_angle : dictionary of angles, optional
        opening angle of the cone the diffuse flux has been generated in;
    extensions : dictionary of solid angles, optional
        solid angle the diffuse flux has been generated in

    Note
    ----
    • for any channel, `diff_angle` superseeds `extensions` if both are given
    • if neither `diff_angle` nor `extensions` is set for a given channel,
      -> assume point source
    """

    # setting defaults because mutable
    diff_angle = diff_angle or {}
    extensions = extensions or {}
    for cl, val in diff_angle.items():
        extensions[cl] = 2 * np.pi * (1 - np.cos(diff_angle[cl])) * u.rad**2

    for cl in events:
        mc_energy = events[cl][irf.mc_energy_name].values * irf.energy_unit

        # event weights for a flat energy distribution
        e_w = mc_energy**generator_gamma[cl] \
            * (e_min_max[cl][1]**(1 - generator_gamma[cl]) -
               e_min_max[cl][0]**(1 - generator_gamma[cl])) / \
              (1 - generator_gamma[cl]) \
            * generator_areas[cl] \
            * (extensions[cl] if (cl in extensions) else 1) \
            * observation_time / n_simulated_events[cl]

        # multiply these flat event weights by the flux to get weights corresponding
        # to the number of expected events from that flux
        events[cl]["weight"] = (e_w * spectra[cl](mc_energy)).si

    return events


def unbinned_wrapper(events):
    return unbinned(
        events, n_simulated_events={'g': irf.meta_data["gamma"]["n_simulated"],
                                    'p': irf.meta_data["proton"]["n_simulated"],
                                    'e': irf.meta_data["electron"]["n_simulated"]},
        generator_areas={'g': np.pi * (irf.meta_data["gamma"]["gen_radius"] * u.m)**2,
                         'p': np.pi * (irf.meta_data["proton"]["gen_radius"] * u.m)**2,
                         'e': np.pi * (irf.meta_data["electron"]["gen_radius"] * u.m)**2},
        observation_time=irf.observation_time,
        spectra={'g': irf.spectra.crab_source_rate,
                 'p': irf.spectra.cr_background_rate,
                 'e': irf.spectra.electron_spectrum},
        e_min_max={'g': (irf.meta_data["gamma"]["e_min"],
                         irf.meta_data["gamma"]["e_max"]) * u.TeV,
                   'p': (irf.meta_data["proton"]["e_min"],
                         irf.meta_data["proton"]["e_max"]) * u.TeV,
                   'e': (irf.meta_data["electron"]["e_min"],
                         irf.meta_data["electron"]["e_max"]) * u.TeV},
        diff_angle={'p': irf.meta_data["proton"]["diff_cone"] * u.deg,
                    'e': irf.meta_data["electron"]["diff_cone"] * u.deg},
        generator_gamma={'g': irf.meta_data["gamma"]["gen_gamma"],
                         'p': irf.meta_data["proton"]["gen_gamma"],
                         'e': irf.meta_data["electron"]["gen_gamma"]})
