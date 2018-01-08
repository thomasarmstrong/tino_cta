import numpy as np
from scipy import optimize

import irf_builder as irf


def cut_and_sensitivity(cuts, events, bin_edges, r_scale,
                        syst_nsim=True, syst_nphy=False):
    """ throw this into a minimiser """
    ga_cut = cuts[0]
    xi_cut = cuts[1]

    cut_events = {}
    for key in events:
        cut_events[key] = events[key][
            (events[key]["gammaness"] > ga_cut) &
            # the background regions are larger to gather more statistics
            (events[key]["off_angle"] < xi_cut * (1 if key == 'g' else r_scale))]

    if syst_nsim and (len(events['g']) < 10 or
                      len(events['g']) < (len(events['p']) + len(events['e'])) * 0.05):
        return 1

    if syst_nphy and (np.sum(events['g']['weight']) < 10 or
                      np.sum(events['g']['weight']) <
                      (np.sum(events['p']['weight']) +
                       np.sum(events['e']['weight'])) * 0.05):
        return 1

    sensitivities = irf.calculate_sensitivity(
        cut_events, bin_edges, alpha=r_scale**-2)

    try:
        return sensitivities["Sensitivity"][0]
    except (KeyError, IndexError):
        return 1


def minimise_sensitivity_per_bin(events, bin_edges, r_scale):

    cut_events = {}
    cut_energies, ga_cuts, xi_cuts = [], [], []
    for elow, ehigh, emid in zip(bin_edges[:-1], bin_edges[1:],
                                 np.sqrt(bin_edges[:-1] * bin_edges[1:])):

        for key in events:
            cut_events[key] = events[key][
                (events[key][irf.reco_energy_name] > elow) &
                (events[key][irf.reco_energy_name] < ehigh)]

        res = optimize.differential_evolution(
            cut_and_sensitivity,
            bounds=[(.5, 1), (0, 0.5)],
            maxiter=2000, popsize=20,
            args=(cut_events,
                  np.array([elow / irf.energy_unit,
                            ehigh / irf.energy_unit]) * irf.energy_unit,
                  r_scale)
        )

        if res.success:
            cut_energies.append(emid.value)
            ga_cuts.append(res.x[0])
            xi_cuts.append(res.x[1])

    return cut_energies, ga_cuts, xi_cuts
