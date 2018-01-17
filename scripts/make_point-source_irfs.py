#!/usr/bin/env python

import sys
from os.path import expandvars
import argparse

import numpy as np
from astropy import units as u
from astropy.table import Table

import pandas as pd

from scipy import interpolate

from matplotlib import pyplot as plt

import irf_builder as irf


def correct_off_angle(data, origin=None):
    import ctapipe.utils.linalg as linalg
    origin = origin or linalg.set_phi_theta(90 * u.deg, 20 * u.deg)

    reco_dirs = linalg.set_phi_theta(data["phi"] * u.deg.to(u.rad),
                                     data["theta"] * u.deg.to(u.rad)).T
    off_angles = np.arccos(np.clip(np.dot(reco_dirs, origin), -1., 1.)) * u.rad
    data["off_angle"] = off_angles.to(u.deg)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--indir',
                    default=expandvars("$CTA_SOFT/tino_cta/data/prod3b/paranal_LND"))
parser.add_argument('--infile', type=str, default="classified_events")
parser.add_argument('--meta_file', type=str, default="meta_data.yml")
parser.add_argument('--r_scale', type=float, default=5.)
parser.add_argument('-k', type=int, default=1, help="order of spline interpolation")

cut_store_group = parser.add_mutually_exclusive_group()
cut_store_group.add_argument('--make_cuts', action='store_true', default=False,
                             help="determines optimal bin-wise gammaness and theta cut "
                             "values and stores them to disk in an astropy table")
cut_store_group.add_argument('--load_cuts', dest='make_cuts', action='store_false',
                             help="loads the gammaness and theta cut values from an "
                             "astropy table from disk")
parser.add_argument('--write_cuts', action='store_true', default=False,
                    help="write cuts as a latex table to disk")

show_plots_group = parser.add_argument_group()
show_plots_group.add_argument('--plot_all', default=False, action='store_true',
                              help="display all plots on screen")
show_plots_group.add_argument('--plot_cuts', default=False, action='store_true',
                              help="display cut-values plots on screen")
show_plots_group.add_argument('--plot_energy', default=False, action='store_true',
                              help="display energy related plots on screen")
show_plots_group.add_argument('--plot_rates', default=False, action='store_true',
                              help="display signal and background rates on screen")
show_plots_group.add_argument('--plot_ang_res', default=False, action='store_true',
                              help="display plots showing angular performance on screen")
show_plots_group.add_argument('--plot_selection', default=False, action='store_true',
                              help="display effective areas, selection efficiencies "
                                   "and number of selected events on screen")
show_plots_group.add_argument('--plot_sens', default=False, action='store_true',
                              help="display sensitivity on screen")

args = parser.parse_args()

irf.r_scale = args.r_scale
irf.alpha = irf.r_scale**-2

# reading the meta data that describes the MC production
irf.meta_data = irf.load_meta_data(f"{args.indir}/{args.meta_file}")

# reading the reconstructed and classified events
all_events = {}
# for mode in ["wave", "tail"]:
for mode in ["wave", "tail"]:
    all_events[mode] = {}
    for c, channel in irf.plotting.channel_map.items():
        all_events[mode][c] = \
            pd.read_hdf(f"{args.indir}/{args.infile}_{channel}_{mode}.h5")

# FUCK FUCK FUCK FUCK
for c in irf.plotting.channel_map:
    correct_off_angle(all_events["wave"][c])


# adding a "weight" column to the data tables
for events in all_events.values():
    irf.make_weights(events)


# # # # # #
# determine optimal bin-by-bin cut values and fit splines to them

cut_energies, ga_cuts, th_cuts = {}, {}, {}

if args.make_cuts:
    print("making cut values")
    for mode, events in all_events.items():
        cut_energies[mode], ga_cuts[mode], th_cuts[mode] = \
            irf.optimise_cuts(events, irf.e_bin_edges, irf.r_scale)

    if args.write_cuts:
        Table([cut_energies[mode], ga_cuts[mode], th_cuts[mode]],
              names=["Energy", "gammaness", "theta"]) \
            .write(filename=f"cut_values_{mode}.tex",
                   # path=args.indir,
                   format="ascii.latex")
else:
    print("loading cut values")
    for mode in all_events:
        cuts = Table.read(f"cut_values_{mode}.tex", format="ascii.latex")
        cut_energies[mode] = cuts["Energy"]
        ga_cuts[mode] = cuts["gammaness"]
        th_cuts[mode] = cuts["theta"]

print("making splines")
spline_ga, spline_xi = {}, {}
for mode in cut_energies:
    spline_ga[mode] = interpolate.splrep(cut_energies[mode], ga_cuts[mode], k=args.k)
    spline_xi[mode] = interpolate.splrep(cut_energies[mode], th_cuts[mode], k=args.k)

    if args.plot_cuts or args.plot_all:
        fig = plt.figure(figsize=(10, 5))
        plt.suptitle(mode)
        for i, (cut_var, spline, ylabel) in enumerate(zip(
                [ga_cuts[mode], th_cuts[mode]],
                [spline_ga[mode], spline_xi[mode]],
                ["gammaness", r"$\Theta_\mathrm{cut} / ^\circ$"])):
            fig.add_subplot(121 + i)
            plt.plot(cut_energies[mode] / u.TeV, cut_var,
                     label="crit. values", ls="", marker="^")
            plt.plot(irf.e_bin_centres_fine / u.TeV,
                     interpolate.splev(irf.e_bin_centres_fine, spline),
                     label="spline fit")

            plt.xlabel("Energy / TeV")
            plt.ylabel(ylabel)
            plt.gca().set_xscale("log")
            plt.legend()

            if i == 0:
                plt.plot(irf.e_bin_centres_fine[[0, -1]], [1, 1],
                         ls="dashed", color="lightgray")
        plt.pause(.1)

# evaluating cuts and add columns with flags
for mode, events in all_events.items():
    for key in events:
        events[key]["pass_gammaness"] = \
            events[key]["gammaness"] > interpolate.splev(events[key]["reco_Energy"],
                                                         spline_ga[mode])
        events[key]["pass_theta"] = \
            events[key]["off_angle"] < (1 if key == 'g' else irf.r_scale) * \
            interpolate.splev(events[key]["reco_Energy"], spline_xi[mode])


# applying the cuts
cut_events = dict(
    (m, irf.event_selection.apply_cuts(e, ["pass_gammaness", "pass_theta"]))
    for m, e in all_events.items())


gamma_events = dict((m, irf.event_selection.apply_cuts(e, ["pass_gammaness"]))
                    for m, e in all_events.items())


# measure and correct for the energy bias
energy_resolution = irf.irfs.energy.get_energy_resolution(cut_events["wave"])
energy_bias = irf.irfs.energy.get_energy_bias(cut_events["wave"])
irf.irfs.energy.correct_energy_bias(cut_events["wave"], energy_bias['g'])

if args.plot_energy or args.plot_all:
    plt.figure(figsize=(10, 5))
    energy_matrix = irf.irfs.energy.get_energy_migration_matrix(cut_events["wave"])
    irf.plotting.plot_energy_migration_matrix(energy_matrix)

    plt.figure(figsize=(10, 5))
    rel_delta_e_reco = irf.irfs.energy.get_rel_delta_e(cut_events["wave"])
    irf.plotting.plot_rel_delta_e(rel_delta_e_reco)

    plt.figure(figsize=(10, 5))
    rel_delta_e_mc = irf.irfs.energy.get_rel_delta_e(cut_events["wave"],
                                                     irf.mc_energy_name)
    irf.plotting.plot_rel_delta_e(rel_delta_e_mc)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xlabel(r"$E_\mathrm{MC}$ / TeV")
        if i == 0:
            ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC}) / E_\mathrm{MC}$")

    plt.figure()
    irf.plotting.plot_energy_resolution(energy_resolution)

    plt.figure()
    irf.plotting.plot_energy_bias(energy_bias)

    plt.figure()
    energy_bias_2 = irf.irfs.energy.get_energy_bias(cut_events["wave"])
    irf.plotting.plot_energy_bias(energy_bias_2)
    plt.title("post e-bias correction")

    plt.figure()
    energy_resolution2 = irf.irfs.energy.get_energy_resolution(cut_events["wave"])
    irf.plotting.plot_energy_resolution(energy_resolution2)
    plt.title("post e-bias correction")


if args.plot_rates or args.plot_all:
    plt.figure()
    energy_rates = irf.irfs.event_rates.get_energy_event_rates(
        cut_events["wave"], th_cuts["wave"])
    irf.plotting.plot_energy_event_rates(energy_rates)


if args.plot_selection or args.plot_all:
    eff_areas, selec_effs, selec_events = \
        irf.irfs.get_effective_areas(cut_events["wave"])
    plt.figure()
    irf.plotting.plot_effective_areas(eff_areas)
    plt.figure()
    irf.plotting.plot_selection_efficiencies(selec_effs)

    plt.figure()
    generator_energies = irf.irfs.get_simulated_energy_distribution(cut_events["wave"])
    irf.plotting.plot_energy_distribution(energies=generator_energies)

    plt.plot(irf.e_bin_centres, selec_events['p'], label="sel. protons")
    plt.plot(irf.e_bin_centres, selec_events['g'], label="sel. gammas")
    plt.legend()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")


if args.plot_ang_res or args.plot_all:
    plt.figure()
    th_sq, bin_e = irf.irfs.angular_resolution.get_theta_square(cut_events["wave"])
    irf.plotting.plot_theta_square(th_sq, bin_e)

    plt.figure()
    xi = irf.irfs.angular_resolution.get_angular_resolution(gamma_events["wave"])
    irf.plotting.plot_angular_resolution(xi)

    plt.figure()
    irf.plotting.plot_angular_resolution_violin(gamma_events["wave"])


sensitivity = {}
for mode, events in cut_events.items():
    sensitivity[mode] = irf.calculate_sensitivity(
        events, irf.e_bin_edges, alpha=irf.alpha, n_draws=1)

if args.plot_sens or args.plot_all:
    plt.figure()
    irf.plotting.plot_crab()
    irf.plotting.plot_reference()
    irf.plotting.plot_sensitivity(sensitivity)


plt.show()
