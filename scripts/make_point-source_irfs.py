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
parser.add_argument('--k', type=int, default=1, help="order of spline interpolation")
parser.add_argument('--plot', default=True, action='store_true',
                    help="display plots on screen")

cut_store_group = parser.add_mutually_exclusive_group()
cut_store_group.add_argument('--make_cuts', action='store_true', default=True,
                             help="determines optimal bin-wise gammaness and theta cut "
                             "values and stores them to disk in an astropy table")
cut_store_group.add_argument('--load_cuts', dest='make_cuts', action='store_false',
                             help="loads the gammaness and theta cut values from an "
                             "astropy table from disk")

args = parser.parse_args()

# reading the meta data that describes the MC production

irf.meta_data = irf.load_meta_data(f"{args.indir}/{args.meta_file}")

# reading the reconstructed and classified events

gammas_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "gamma", "wave"), "reco_events")
proton_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "proton", "wave"), "reco_events")
electr_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "electron", "wave"), "reco_events")

# FUCK FUCK FUCK FUCK
correct_off_angle(gammas_w_o)
correct_off_angle(proton_w_o)
correct_off_angle(electr_w_o)

events_w = {"reco": {'g': gammas_w_o, 'p': proton_w_o, 'e': electr_w_o}}

gammas_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "gamma", "tail"), "reco_events")
proton_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "proton", "tail"), "reco_events")
electr_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
    args.indir, args.infile, "electron", "tail"), "reco_events")

events_t = {"reco": {'g': gammas_t_o, 'p': proton_t_o, 'e': electr_t_o}}


irf.make_weights(events_w["reco"])
irf.make_weights(events_t["reco"])

# determine optimal bin-by-bin cut values and fit splines to them

cut_energies, ga_cuts, xi_cuts = {}, {}, {}

if args.make_cuts:
    print("making cut values")
    for mode in ["wave", "tail"]:
        cut_energies[mode], ga_cuts[mode], xi_cuts[mode] = \
            irf.optimise_cuts(events_w["reco"], irf.e_bin_edges, args.r_scale)
    cuts = Table([cut_energies[mode], ga_cuts[mode], xi_cuts[mode]],
                 names=["Energy", "gammaness", "xi"])
    cuts.write(filename=f"cut_values_{mode}.tex",
               # path=args.indir,
               format="ascii.latex")
else:
    print("loading cut values")
    for mode in ["wave", "tail"]:
        cuts = Table.read(f"cut_values_{mode}.tex", format="ascii.latex")
        cut_energies[mode] = cuts["Energy"]
        ga_cuts[mode] = cuts["gammaness"]
        xi_cuts[mode] = cuts["xi"]

print("making splines")
spline_ga, spline_xi = {}, {}
for mode in cut_energies:
    spline_ga[mode] = interpolate.splrep(cut_energies[mode], ga_cuts[mode], k=args.k)
    spline_xi[mode] = interpolate.splrep(cut_energies[mode], xi_cuts[mode], k=args.k)

#     fig = plt.figure(figsize=(10, 5))
#     plt.suptitle(mode)
#     for i, (cut_var, spline, ylabel) in enumerate(zip(
#             [ga_cuts[mode], xi_cuts[mode]],
#             [spline_ga[mode], spline_xi[mode]],
#             ["gammaness", "xi / degree"])):
#         fig.add_subplot(121 + i)
#         plt.plot(cut_energies[mode] / u.TeV, cut_var,
#                  label="crit. values", ls="", marker="^")
#         plt.plot(irf.e_bin_centres_fine / u.TeV,
#                  interpolate.splev(irf.e_bin_centres_fine, spline),
#                  label="spline fit")
#
#         plt.xlabel("Energy / TeV")
#         plt.ylabel(ylabel)
#         plt.gca().set_xscale("log")
#         plt.legend()
#
#         if i == 0:
#             plt.plot(irf.e_bin_centres_fine[[0, -1]], [1, 1],
#                      ls="dashed", color="lightgray")
#
# if args.plot:
#     plt.pause(.1)

# applying cuts consecutively

from_step = "reco"
next_step = "gammaness"
for mode, events in [("wave", events_w), ("tail", events_t)]:
    events[next_step] = {}
    for key in events[from_step]:
        events[next_step][key] = events[from_step][key][
            events[from_step][key]["gammaness"] >
            interpolate.splev(events[from_step][key]["reco_Energy"], spline_ga[mode])]

from_step = "gammaness"
next_step = "theta"
for mode, events in [("wave", events_w), ("tail", events_t)]:
    events[next_step] = {}
    for key in events[from_step]:
        events[next_step][key] = events[from_step][key][
            events[from_step][key]["off_angle"] < (1 if key == 'g' else args.r_scale) *
            interpolate.splev(events[from_step][key]["reco_Energy"], spline_xi[mode])]

sensitivity = {}
for mode, events in [("wave", events_w["theta"]),
                     ("tail", events_t["theta"])]:
    sensitivity[mode] = irf.calculate_sensitivity(
        events, irf.e_bin_edges, alpha=args.r_scale**-2)

plt.figure()
irf.plotting.plot_crab()
irf.plotting.plot_reference()
irf.plotting.plot_sensitivity(sensitivity)


# plt.figure()
# eff_areas = irf.irfs.get_effective_areas(events_w["reco"])
# irf.plotting.plot_effective_areas(eff_areas)


# plt.figure()
# th_sq, bin_e = irf.irfs.angular_resolution.get_theta_square(events_w["theta"])
# irf.plotting.plot_theta_square(th_sq, bin_e)


# plt.figure()
# xi = irf.irfs.angular_resolution.get_angular_resolution(events_w["gammaness"])
# irf.plotting.plot_angular_resolution(xi)

# plt.figure()
# irf.plotting.plot_angular_resolution_violin(events_w["gammaness"])

plt.show()
