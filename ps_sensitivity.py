#!/usr/bin/env python3
import glob
import numpy as np

# PyTables
import tables as tb
# pandas data frames
import pandas as pd

from astropy.table import Table
from astropy import units as u

from itertools import chain

from scipy.optimize import curve_fit

from helper_functions import *

from ctapipe.analysis.sensitivity import (SensitivityPointSource, e_minus_2,
                                          crab_source_rate, cr_background_rate)

import matplotlib.pyplot as plt
# plt.style.use('seaborn-poster')
# plt.style.use('t_slides')
plt.rc('text', usetex=True)


# define edges to sort events in
e_bin_edges = np.logspace(-2, np.log10(330), 20)*u.TeV
e_bin_centres = (e_bin_edges[:-1] + e_bin_edges[1:])/2


# MC energy ranges:
# gammas: 0.003 to 330 TeV
# proton: 0.004 to 600 TeV
edges_gammas = np.logspace(np.log10(0.003), np.log10(330), 28) * u.TeV
edges_proton = np.logspace(np.log10(0.004), np.log10(600), 30) * u.TeV
sensitivity_energy_bin_edges = np.logspace(-2, 2.5, 17)*u.TeV


# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg*u.cm**2*u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# scale MC events to this reference time
observation_time = 50*u.h


# ASTRI:
# NReuse_Gammas = 10
# NReuse_Proton = 20
#
# NGammas_per_File = 5000 * NReuse_Gammas
# NProton_per_File = 5000 * NReuse_Proton
#
# NGammas_simulated = NGammas_per_File * (498-14)
# NProton_simulated = NProton_per_File * (6998-100)


NGammas_per_File = 20000
NProton_per_File = 100000
NGammas_simulated = NGammas_per_File * (5000-30)
NProton_simulated = NProton_per_File * (40000-30)


def fitfunc(x, a, b, c, d, e=0, f=0, g=0, h=0):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7


def fitfunc_log(x, a, b, c, d, e=0, f=0, g=0, h=0):
    x = np.log10(x)
    return fitfunc(x, a, b, c, d, e, f, g, h)


def percentiles(values, bin_values, bin_edges, percentile):
    percentiles_binned = \
        np.squeeze(np.full((len(bin_edges)-1, len(values.shape)), np.inf))
    for i, (bin_l, bin_h) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        try:
            percentiles_binned[i] = \
                np.percentile(values[(bin_values > bin_l) &
                                     (bin_values < bin_h)], percentile)
        except IndexError:
            pass
    return percentiles_binned.T


def main(xi_percentile={'w': 68, 't': 68}, xi_on_scale=1, xi_off_scale=20,
         ga_percentile={'w': 99, 't': 99}):

    def selection_mask(event_table, gammaness=.75, ntels=3):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness))

    apply_cuts = True
    gammaness_wave = .95
    gammaness_tail = .95

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "gamma", "wave"), "reco_events")

    proton_w_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "proton", "wave"), "reco_events")

    gammas_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "gamma", "tail"), "reco_events")

    proton_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "proton", "tail"), "reco_events")

    print("\n")
    print("gammas present (wavelets):", len(gammas_w_o))
    print("proton present (wavelets):", len(proton_w_o))
    print()
    print("gammas present (tailcuts):", len(gammas_t_o))
    print("proton present (tailcuts):", len(proton_t_o))

    #  ######      ###    ##     ## ##     ##    ###    ##    ##  ######
    # ##    ##    ## ##   ###   ### ###   ###   ## ##   ###   ## ##    ##
    # ##         ##   ##  #### #### #### ####  ##   ##  ####  ## ##
    # ##   #### ##     ## ## ### ## ## ### ## ##     ## ## ## ##  ######
    # ##    ##  ######### ##     ## ##     ## ######### ##  ####       ##
    # ##    ##  ##     ## ##     ## ##     ## ##     ## ##   ### ##    ##
    #  ######   ##     ## ##     ## ##     ## ##     ## ##    ##  ######

    g_cuts_w = percentiles(
            proton_w_o["gammaness"], proton_w_o["reco_Energy"],
            e_bin_edges.value, ga_percentile['w'])
    g_cuts_t = percentiles(
            proton_t_o["gammaness"], proton_t_o["reco_Energy"],
            e_bin_edges.value, ga_percentile['t'])
    popt_g_w, pcov_g_w = curve_fit(
            fitfunc_log, e_bin_centres[g_cuts_w != np.inf],
            g_cuts_w[g_cuts_w != np.inf])
    popt_g_t, pcov_g_t = curve_fit(
            fitfunc_log, e_bin_centres[g_cuts_t != np.inf],
            g_cuts_t[g_cuts_t != np.inf])

    if False:
        plt.figure()
        plt.semilogx(e_bin_centres, gammaness_cuts_w, ls="", marker="^",
                     label="crit. values -- {} \%".format(ga_percentile['w']))
        plt.semilogx(e_bin_centres, fitfunc_log(e_bin_centres.value, *popt_g_w),
                     ls="-", marker="", label="poly. fit")
        plt.semilogx(e_bin_centres[[0, -1]], [gammaness_wave, gammaness_wave],
                     ls=":", label="const. cut -- {}".format(gammaness_wave))
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel("gammaness")
        plt.legend()
        plt.pause(.1)

    gammas_w_g = gammas_w_o[
            gammas_w_o["gammaness"] > fitfunc_log(gammas_w_o["reco_Energy"],
                                                  *popt_g_w)]
    proton_w_g = proton_w_o[
            proton_w_o["gammaness"] > fitfunc_log(proton_w_o["reco_Energy"],
                                                  *popt_g_w)]

    gammas_t_g = gammas_t_o[
            gammas_t_o["gammaness"] > fitfunc_log(gammas_t_o["reco_Energy"],
                                                  *popt_g_w)]
    proton_t_g = proton_t_o[
            proton_t_o["gammaness"] > fitfunc_log(proton_t_o["reco_Energy"],
                                                  *popt_g_w)]

    # ##     ## ####     ######  ##     ## ########
    #  ##   ##   ##     ##    ## ##     ##    ##
    #   ## ##    ##     ##       ##     ##    ##
    #    ###     ##     ##       ##     ##    ##
    #   ## ##    ##     ##       ##     ##    ##
    #  ##   ##   ##     ##    ## ##     ##    ##
    # ##     ## ####     ######   #######     ##

    xi_cuts_w = percentiles(gammas_w_g["off_angle"], gammas_w_g["reco_Energy"],
                            e_bin_edges.value, xi_percentile['w'])
    xi_cuts_t = percentiles(gammas_t_g["off_angle"], gammas_t_g["reco_Energy"],
                            e_bin_edges.value, xi_percentile['t'])

    # fit a polynomial to the cut values in xi
    popt_w, pcov_w = curve_fit(fitfunc_log,
                               e_bin_centres[xi_cuts_w != np.inf].value,
                               xi_cuts_w[xi_cuts_w != np.inf])
    popt_t, pcov_t = curve_fit(fitfunc_log,
                               e_bin_centres[xi_cuts_t != np.inf].value,
                               xi_cuts_t[xi_cuts_t != np.inf])

    if False:
        plt.figure()
        plt.loglog(e_bin_centres, xi_cuts_t,
                   color="darkorange", marker="^", ls="",
                   label="MC tail -- {} %".format(xi_percentile['t']))
        plt.loglog(e_bin_centres, xi_cuts_w,
                   color="darkred", marker="^", ls="",
                   label="MC wave -- {} %".format(xi_percentile['w']))
        plt.loglog(e_bin_centres, fitfunc_log(e_bin_centres.value, *popt_t),
                   color="darkorange", label="fit tail")
        plt.loglog(e_bin_centres, fitfunc_log(e_bin_centres.value, *popt_w),
                   color="darkred", label="fit wave")
        plt.title("on-region size")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$\xi_\mathrm{cut} / ^\circ$")
        plt.grid()
        plt.legend()
        plt.pause(.1)

    if apply_cuts:
        gammas_w_rcut = gammas_w_g[
                gammas_w_g["off_angle"]
                < fitfunc_log(gammas_w_g["reco_Energy"], *popt_w)*xi_on_scale]

        proton_w_rcut = proton_w_g[
                proton_w_g["off_angle"]
                < fitfunc_log(proton_w_g["reco_Energy"], *popt_w)*xi_off_scale]

        gammas_t_rcut = gammas_t_g[
                gammas_t_g["off_angle"]
                < fitfunc_log(gammas_t_g["reco_Energy"], *popt_t)*xi_on_scale]

        proton_t_rcut = proton_t_g[
                proton_t_g["off_angle"]
                < fitfunc_log(proton_t_g["reco_Energy"], *popt_t)*xi_off_scale]

    print("\n")
    print("gammas selected (wavelets):", len(gammas_w_rcut))
    print("proton selected (wavelets):", len(proton_w_rcut))
    print()
    print("gammas selected (tailcuts):", len(gammas_t_rcut))
    print("proton selected (tailcuts):", len(proton_t_rcut))

    SensCalc_w = SensitivityPointSource(
            reco_energies={'g': gammas_w_g['reco_Energy'].values*u.TeV,
                           'p': proton_w_g['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas_w_g['MC_Energy'].values*u.TeV,
                         'p': proton_w_g['MC_Energy'].values*u.TeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    SensCalc_w.get_effective_areas(
            n_simulated_events={'g': NGammas_simulated,
                                'p': NProton_simulated},
            generator_spectra={'g': e_minus_2,
                               'p': e_minus_2},
            generator_areas={'g': np.pi * (2500*u.m)**2,
                             'p': np.pi * (3000*u.m)**2},
    )

    SensCalc_w.generate_event_weights(
            n_simulated_events={'g': NGammas_simulated,
                                'p': NProton_simulated},
            generator_areas={'g': np.pi * (2500*u.m)**2,
                             'p': np.pi * (3000*u.m)**2},
            observation_time=observation_time,
            spectra={'g': crab_source_rate,
                     'p': cr_background_rate},
            e_min_max={"g": (0.003, 330)*u.TeV,
                       "p": (0.004, 600)*u.TeV},
            extensions={'p': 10 * u.deg},
            generator_gamma={"g": 2, "p": 2})

    sensitivities_w = SensCalc_w.get_sensitivity(
            alpha=(xi_on_scale/xi_off_scale)**2,
            sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    # sensitvity for tail cuts
    SensCalc_t = SensitivityPointSource(
            reco_energies={'g': gammas_t_rcut['reco_Energy'].values*u.TeV,
                           'p': proton_t_rcut['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas_t_rcut['MC_Energy'].values*u.TeV,
                         'p': proton_t_rcut['MC_Energy'].values*u.TeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    SensCalc_t.get_effective_areas(
            n_simulated_events={'g': NGammas_simulated,
                                'p': NProton_simulated},
            generator_spectra={'g': e_minus_2,
                               'p': e_minus_2},
            generator_areas={'g': np.pi * (2500*u.m)**2,
                             'p': np.pi * (3000*u.m)**2},
    )
    SensCalc_t.generate_event_weights(
            n_simulated_events={'g': NGammas_simulated,
                                'p': NProton_simulated},
            generator_areas={'g': np.pi * (2500*u.m)**2,
                             'p': np.pi * (3000*u.m)**2},
            observation_time=observation_time,
            spectra={'g': crab_source_rate,
                     'p': cr_background_rate},
            e_min_max={"g": (0.003, 330)*u.TeV,
                       "p": (0.004, 600)*u.TeV},
            extensions={'p': 10 * u.deg},
            generator_gamma={"g": 2, "p": 2})

    sensitivities_t = SensCalc_t.get_sensitivity(
            alpha=(xi_on_scale/xi_off_scale)**2,
            sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    print("\n")
    print("gammas expected (wavelets):", np.sum(SensCalc_w.event_weights["g"]))
    print("proton expected (wavelets):", np.sum(SensCalc_w.event_weights["p"]))
    print()
    print("gammas expected (tailcuts):", np.sum(SensCalc_t.event_weights["g"]))
    print("proton expected (tailcuts):", np.sum(SensCalc_t.event_weights["p"]))

    # make_performance_plots(gammas_w_o, proton_w_o, gammas_t_o, proton_t_o)
    # make_performance_plots(gammas_w_g, proton_w_g, gammas_t_g, proton_t_g)
    make_performance_plots(gammas_w_rcut, proton_w_rcut, gammas_t_rcut, proton_t_rcut)

    # show_gammaness(gammas_w_rcut, proton_w_rcut, "wave")
    # show_gammaness(gammas_t_rcut, proton_t_rcut, "tail")

    make_sensitivity_plots(SensCalc_w, sensitivities_w,
                           SensCalc_t, sensitivities_t)


# ########  ##        #######  ########  ######
# ##     ## ##       ##     ##    ##    ##    ##
# ##     ## ##       ##     ##    ##    ##
# ########  ##       ##     ##    ##     ######
# ##        ##       ##     ##    ##          ##
# ##        ##       ##     ##    ##    ##    ##
# ##        ########  #######     ##     ######

def make_sensitivity_plots(SensCalc, sensitivities, SensCalc_t, sensitivities_t):
    bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.
    bin_centres_p = (edges_proton[1:]+edges_proton[:-1])/2.

    bin_widths_g = np.diff(edges_gammas.value)
    bin_widths_p = np.diff(edges_proton.value)

    if args.verbose:
        # plot MC generator spectrum and selected spectrum
        # plt.figure()
        # plt.subplot(121)
        # plt.bar(bin_centres_g.value,
        #         SensCalc_t.generator_energy_hists['g'], label="generated",
        #         align="center", width=bin_widths_g)
        # plt.bar(bin_centres_g.value,
        #         SensCalc_t.selected_events['g'], label="selected",
        #         align="center", width=bin_widths_g)
        # plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        # plt.ylabel("number of events")
        # plt.gca().set_xscale("log")
        # plt.gca().set_yscale("log")
        # plt.title("gammas -- tailcuts")
        # plt.legend()
        #
        # plt.subplot(122)
        # plt.bar(bin_centres_p.value,
        #         SensCalc_t.generator_energy_hists['p'], label="generated",
        #         align="center", width=bin_widths_p)
        # plt.bar(bin_centres_p.value,
        #         SensCalc_t.selected_events['p'], label="selected",
        #         align="center", width=bin_widths_p)
        # plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        # plt.ylabel("number of events")
        # plt.gca().set_xscale("log")
        # plt.gca().set_yscale("log")
        # plt.title("protons -- tailcuts")
        # plt.legend()

        # plot the number of expected events in each energy bin
        plt.figure()
        plt.bar(
                bin_centres_p.value,
                SensCalc.exp_events_per_energy_bin['p'], label="proton",
                align="center", width=np.diff(edges_proton.value), alpha=.75)
        plt.bar(
                bin_centres_g.value,
                SensCalc.exp_events_per_energy_bin['g'], label="gamma",
                align="center", width=np.diff(edges_gammas.value), alpha=.75)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")

        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        plt.ylabel("expected events in {}".format(observation_time))
        plt.legend()

        # plot effective area
        plt.figure()  # figsize=(16, 8))
        plt.suptitle("Effective Areas")
        # plt.subplot(121)
        plt.loglog(
            bin_centres_g,
            SensCalc.effective_areas['g'], label="wavelets", color="darkred", marker="^")
        # plt.loglog(
        #     bin_centres_g,
        #     SensCalc_t.effective_areas['g'], label="tailcuts")
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
        plt.title("gammas")
        plt.legend()

        # plt.subplot(122)
        # plt.loglog(
        #     bin_centres_p,
        #     SensCalc.effective_areas['p'], label="wavelets")
        # plt.loglog(
        #     bin_centres_p,
        #     SensCalc_t.effective_areas['p'], label="tailcuts")
        # plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_p.unit)+"}$")
        # plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
        # plt.title("protons")
        # plt.legend()

        # plot the angular distance of the reconstructed shower direction
        # from the pseudo-source

        # figure = plt.figure()
        # bins = 60
        #
        # plt.subplot(211)
        # plt.hist([proton_t['off_angle']**2,
        #           gammas_t["off_angle"]**2],
        #          weights=[weights_t['p'], weights_t['g']],
        #          rwidth=1, stacked=True,
        #          range=(0, .3), label=("protons", "gammas"),
        #          log=False, bins=bins)
        # plt.xlabel(r"$(\vartheta/^\circ)^2$")
        # plt.ylabel("expected events in {}".format(observation_time))
        # plt.xlim([0, .3])
        # plt.legend(loc="upper right", title="tailcuts")
        #
        # plt.subplot(212)
        # plt.hist([proton['off_angle']**2,
        #           gammas["off_angle"]**2],
        #          weights=[weights['p'], weights['g']],
        #          rwidth=1, stacked=True,
        #          range=(0, .3), label=("protons", "gammas"),
        #          log=False, bins=bins)
        # plt.xlabel(r"$(\vartheta/^\circ)^2$")
        # plt.ylabel("expected events in {}".format(observation_time))
        # plt.xlim([0, .3])
        # plt.legend(loc="upper right", title="wavelets")
        # plt.tight_layout()
        #
        # if args.write:
        #     save_fig("plots/theta_square")

    # the point-source sensitivity binned in energy

    plt.figure()

    # draw the crab flux as a reference
    crab_bins = np.logspace(-2, 2.5, 17)
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                * (crab_bins*u.TeV.to(u.erg))**2),
               color="red", ls="dashed", label="Crab Nebula")
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                * (crab_bins*u.TeV.to(u.erg))**2)/10,
               color="red", ls="dashed", alpha=.66, label="Crab Nebula / 10")
    plt.loglog(crab_bins,
               (crab_source_rate(crab_bins*u.TeV).to(flux_unit)
                * (crab_bins*u.TeV.to(u.erg))**2)/100,
               color="red", ls="dashed", alpha=.33, label="Crab Nebula / 100")

    # some semi-official line to compare
    ref_loge, ref_sens = *(np.array([
            (-1.8, 6.87978e-11), (-1.6, 1.87765e-11),
            (-1.4, 7.00645e-12), (-1.2, 1.77677e-12), (-1.0, 8.19263e-13),
            (-0.8, 4.84879e-13), (-0.6, 3.00256e-13), (-0.4, 2.07787e-13),
            (-0.2, 1.4176e-13), (0.0, 1.06069e-13), (0.2, 8.58209e-14),
            (0.4, 6.94294e-14), (0.6, 6.69301e-14), (0.8, 7.61169e-14),
            (1.0, 7.13895e-14), (1.2, 9.49376e-14), (1.4, 1.25208e-13),
            (1.6, 1.91209e-13), (1.8, 3.11611e-13), (2.0, 4.80354e-13)]).T),
    plt.loglog(10**ref_loge,
               ((ref_sens)*(u.erg*u.cm**2*u.s)**(-1)).to(flux_unit),
               marker="s", color="black", ms=3, linewidth=1,
               label="reference")

    sens_low, sens_up = (
        (sensitivities["Sensitivity"] -
         sensitivities["Sensitivity_low"]).to(flux_unit) *
        sensitivities["Energy"].to(u.erg)**2,
        (sensitivities["Sensitivity_up"] -
         sensitivities["Sensitivity"]).to(flux_unit) *
        sensitivities["Energy"].to(u.erg)**2)

    plt.errorbar(
        sensitivities["Energy"],
        (sensitivities["Sensitivity"].to(flux_unit) *
         sensitivities["Energy"].to(u.erg)**2).value,
        (sens_low.value, sens_up.value),
        color="darkred",
        marker="s",
        label="wavelets")
    # plt.semilogy(
    #     sensitivities["Energy"].to(energy_unit),
    #     (sensitivities["Sensitivity_base"].to(flux_unit) *
    #      sensitivities["Energy"].to(u.erg)**2),
    #     color="darkgreen",
    #     marker="^",
    #     # ls="",
    #     label="wavelets (no upscale)")

    # sens_t_low, sens_t_up = (
    #     (sensitivities_t["Sensitivity"] -
    #      sensitivities_t["Sensitivity_low"]).to(flux_unit) *
    #     sensitivities_t["Energy"].to(u.erg)**2,
    #     (sensitivities_t["Sensitivity_up"] -
    #      sensitivities_t["Sensitivity"]).to(flux_unit) *
    #     sensitivities_t["Energy"].to(u.erg)**2)
    #
    # plt.errorbar(
    #     sensitivities_t["Energy"].to(energy_unit).value,
    #     (sensitivities_t["Sensitivity"].to(flux_unit) *
    #      sensitivities_t["Energy"].to(u.erg)**2).value,
    #     (sens_low.value, sens_up.value),
    #     color="C0",
    #     marker="s",
    #     label="tailcuts")
    # plt.semilogy(
    #     sensitivities_t["Energy"].to(energy_unit),
    #     (sensitivities_t["Sensitivity_base"].to(flux_unit) *
    #      sensitivities_t["Energy"].to(u.erg)**2),
    #     color="darkorange",
    #     marker="^",
    #     # ls="",
    #     label="tailcuts (no upscale)")

    plt.legend(title="Obsetvation Time: {}".format(observation_time), loc=1)
    plt.xlabel(r'$E_\mathrm{reco}$' + ' / {:latex}'.format(energy_unit))
    plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
    plt.gca().set_xscale("log")
    plt.grid()
    plt.xlim([1e-2, 2e2])
    # plt.ylim([5e-15, 5e-10])

    # plot the sensitivity ratios
    # plt.figure()
    # plt.semilogx(sensitivities_t["Energy"].to(energy_unit),
    #              (sensitivities["Sensitivity_base"].to(flux_unit) *
    #               sensitivities["Energy"].to(u.erg)**2)[1:] /
    #              (sensitivities_t["Sensitivity_base"].to(flux_unit) *
    #               sensitivities_t["Energy"].to(u.erg)**2),
    #              label=r"Sens$_{wave}$ / Sens$_{tail}$"
    #              )
    # plt.legend()
    # plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[[0, -1]],
    #              [1, 1], ls="--", color="gray")
    # plt.xlim(sensitivities_t["Energy"].to(energy_unit)[[0, -1]].value)
    # plt.ylim([.25, 1.1])
    # plt.xlabel('E / {:latex}'.format(energy_unit))
    # plt.ylabel("ratio")

    # plot a sky image of the events
    # useless since too few MC background events left
    if False:
        fig2 = plt.figure()
        plt.hexbin(
            [(ph-180)*np.sin(th*u.deg) for
                ph, th in zip(chain(gammas['phi'], proton['phi']),
                              chain(gammas['theta'], proton['theta']))],
            [a for a in chain(gammas['theta'], proton['theta'])],
            gridsize=41, extent=[-2, 2, 18, 22],
            C=[a for a in chain(weights['g'], weights['p'])],
            bins='log'
            )
        plt.colorbar().set_label("log(Number of Events)")
        plt.axes().set_aspect('equal')
        plt.xlabel(r"$\sin(\vartheta) \cdot (\varphi-180) / ${:latex}"
                   .format(angle_unit))
        plt.ylabel(r"$\vartheta$ / {:latex}".format(angle_unit))
        if args.write:
            save_fig("plots/skymap")


def show_gammaness(gammas, proton, suptitle=None):

    gamm_bins = np.linspace(0, 1, 101)
    NTels_bins = np.linspace(0, 50, 21)[:]
    energy_bins = np.linspace(-2, 2.5, 16)[1:]
    energy_bin_centres = (energy_bins[:-1]+energy_bins[1:])/2

    # gammaness vs. number of telescopes

    gamm_vs_ntel_g = np.histogram2d(gammas["NTels_reco"], gammas["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_p = np.histogram2d(proton["NTels_reco"], proton["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_g = gamm_vs_ntel_g / gamm_vs_ntel_g.sum(axis=0)
    gamm_vs_ntel_p = gamm_vs_ntel_p / gamm_vs_ntel_p.sum(axis=0)

    fig = plt.figure()
    ax1 = plt.subplot(121)
    im = ax1.imshow(np.sqrt(gamm_vs_ntel_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel("NTels")
    ax1.set_ylabel("gammaness")

    ax2 = plt.subplot(122, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_ntel_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel("NTels")

    cb = fig.colorbar(im, ax=[ax1, ax2], label="sqrt(event fraction per NTels-row)")

    if suptitle:
        plt.suptitle(suptitle)

    #
    # gammaness vs. reconstructed energy

    gamm_vs_e_reco_g = np.histogram2d(
            np.log10(gammas["reco_Energy"]), gammas["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_p = np.histogram2d(
            np.log10(proton["reco_Energy"]), proton["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_g = gamm_vs_e_reco_g / gamm_vs_e_reco_g.sum(axis=0)
    gamm_vs_e_reco_p = gamm_vs_e_reco_p / gamm_vs_e_reco_p.sum(axis=0)

    fig = plt.figure()
    ax1 = plt.subplot(121)
    im = ax1.imshow(np.sqrt(gamm_vs_e_reco_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax1.set_ylabel("gammaness")

    ax2 = plt.subplot(122, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_e_reco_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel(r"$E_\mathrm{reco}$ / TeV")

    cb = fig.colorbar(im, ax=[ax1, ax2], label="sqrt(event fraction per E-row)")

    if suptitle:
        plt.suptitle(suptitle)

    # # same as a wireframe plot
    #
    # from mpl_toolkits.mplot3d import axes3d
    # xv, yv = np.meshgrid(energy_bins, gamm_bins)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe((xv[1:, 1:]+xv[:-1, :-1])/2,
    #                   (yv[1:, 1:]+yv[:-1, :-1])/2, gamm_vs_e_reco_g,
    #                   color="darkred", label="gamma")
    # ax.plot_wireframe((xv[1:, 1:]+xv[:-1, :-1])/2,
    #                   (yv[1:, 1:]+yv[:-1, :-1])/2, gamm_vs_e_reco_p,
    #                   color="darkorange", label="proton")
    # ax.set_xlabel(r"$\log_{10}(E_\mathrm{reco})$ / TeV")
    # ax.set_ylabel("gammaness")
    # ax.set_zlabel("event fraction per E-row")
    # plt.legend()


def make_performance_plots(gammas_w, proton_w,
                           gammas_t, proton_t):

    fig, axes = plt.subplots(1, 2)
    n_tel_max = 50  # np.max(gammas_w["NTels_reco"])
    # plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)
    plot_hex_and_violin(gammas_w["NTels_reco"], np.log10(gammas_w["off_angle"]),
                        np.arange(0, n_tel_max+1, 5),
                        xlabel=r"$N_\mathrm{Tels}$",
                        ylabel=r"$\log_{10}(\xi / ^\circ)$",
                        do_hex=False, axis=axes[0],
                        extent=[0, n_tel_max, -3, 0])
    plot_hex_and_violin(np.log10(gammas_w["reco_Energy"]),
                        np.log10(gammas_w["off_angle"]),
                        np.linspace(-1, 3, 17),
                        xlabel=r"$\log_{10}(E_\mathrm{reco}$ / TeV)",
                        ylabel=r"$\log_{10}(\xi / ^\circ)$",
                        v_padding=0.015, axis=axes[1], extent=[-.5, 2.5, -3, 0])
    plt.suptitle("wavelet")

    xi_68_gw = percentiles(gammas_w["off_angle"], gammas_w["reco_Energy"],
                           e_bin_edges.value, 68)
    xi_68_pw = percentiles(proton_w["off_angle"], proton_w["reco_Energy"],
                           e_bin_edges.value, 68)
    xi_68_gt = percentiles(gammas_t["off_angle"], gammas_t["reco_Energy"],
                           e_bin_edges.value, 68)
    xi_68_pt = percentiles(proton_t["off_angle"], proton_t["reco_Energy"],
                           e_bin_edges.value, 68)

    plt.figure()
    # plt.semilogx(e_bin_centres, xi_68_gt,
    #              color="darkorange", marker="^", ls="-",
    #              label="gamma -- tail")
    # plt.semilogx(e_bin_centres, xi_68_pt,
    #              color="darkorange", marker="o", ls=":",
    #              label="proton -- tail")
    plt.semilogx(e_bin_centres, xi_68_gw,
                 color="darkred", marker="^", ls="-",
                 label="gamma -- wave")
    # plt.semilogx(e_bin_centres, xi_68_pw,
    #              color="darkred", marker="o", ls=":",
    #              label="proton -- wave")
    plt.title("angular resolution")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$\xi_\mathrm{68} / ^\circ$")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.legend()

    plt.pause(.1)

    # MC Energy vs. reco Energy 2D histograms
    fig = plt.figure()
    ax = plt.subplot(121)
    counts_g, _, _ = np.histogram2d(gammas_w["MC_Energy"],
                                    gammas_w["reco_Energy"],
                                    bins=(e_bin_edges, e_bin_edges))
    ax.pcolormesh(e_bin_edges.value, e_bin_edges.value, counts_g)
    plt.plot(e_bin_edges.value[[0, -1]], e_bin_edges.value[[0, -1]],
             color="darkgreen")
    plt.title("gamma")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$E_\mathrm{MC}$ / TeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid()

    ax = plt.subplot(122)
    counts_p, _, _ = np.histogram2d(proton_w["MC_Energy"],
                                    proton_w["reco_Energy"],
                                    bins=(e_bin_edges, e_bin_edges))
    ax.pcolormesh(e_bin_edges.value, e_bin_edges.value, counts_p)
    plt.plot(e_bin_edges.value[[0, -1]], e_bin_edges.value[[0, -1]],
             color="darkgreen")
    plt.title("proton")
    plt.xlabel("$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$E_\mathrm{MC}$ / TeV")
    plt.suptitle("wavelet")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid()
    # plt.subplots_adjust(top=.90, bottom=.11, left=.12, right=.90,
    #                     hspace=.20, wspace=.31)

    plt.pause(.1)

    # energy resolution as 68th percentile of the relative reconstructed error binned in
    # reconstructed energy
    rel_DeltaE_w = np.abs(gammas_w["reco_Energy"] -
                          gammas_w["MC_Energy"])/gammas_w["reco_Energy"]
    DeltaE68_w_ebinned = percentiles(rel_DeltaE_w, gammas_w["reco_Energy"],
                                     e_bin_edges.value, 68)

    rel_DeltaE_t = np.abs(gammas_t["reco_Energy"] -
                          gammas_t["MC_Energy"])/gammas_t["reco_Energy"]
    DeltaE68_t_ebinned = percentiles(rel_DeltaE_t, gammas_t["reco_Energy"],
                                     e_bin_edges.value, 68)

    plt.figure()
    plt.plot(e_bin_centres, DeltaE68_t_ebinned, label="gamma -- tail",
             marker='^', color="darkorange")
    plt.plot(e_bin_centres, DeltaE68_w_ebinned, label="gamma -- wave",
             marker='^', color="darkred")
    plt.title("Energy Resolution")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$")
    plt.gca().set_xscale("log")
    plt.grid()
    plt.legend()

    plt.pause(.1)

    # Ebias as median of 1-E_reco/E_MC
    Ebias_w = 1 - (gammas_w["reco_Energy"]/gammas_w["MC_Energy"])
    Ebias_w_medians = percentiles(Ebias_w, gammas_w["reco_Energy"],
                                  e_bin_edges.value, 50)
    Ebias_t = 1 - (gammas_t["reco_Energy"]/gammas_t["MC_Energy"])
    Ebias_t_medians = percentiles(Ebias_t, gammas_t["reco_Energy"],
                                  e_bin_edges.value, 50)

    plt.figure()
    plt.plot(e_bin_centres, Ebias_t_medians, label="gamma -- tail",
             marker='^', color="darkorange")
    plt.plot(e_bin_centres, Ebias_w_medians, label="gamma -- wave",
             marker='^', color="darkred")
    plt.title("Energy Bias")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$(1 - E_\mathrm{reco}/E_\mathrm{MC})_{50}$")
    plt.ylim([-0.2, .3])
    plt.gca().set_xscale("log")
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    np.random.seed(19)

    parser = make_argparser()
    parser.add_argument('--infile', type=str, default="classified_events")
    args = parser.parse_args()

    main(xi_percentile={'w': 68, 't': 68},
         ga_percentile={'w': 99, 't': 99})

    # for xi in np.arange(50, 90, 3):
    #     for ga in 100*(1-np.logspace(-3, 0, 10)):
    # for xi in [50, 68]:
    #     for ga in [90, 95, 99]:
    #         main(xi_percentile={'w': xi, 't': xi},
    #              ga_percentile={'w': ga, 't': ga})
    #         plt.suptitle("{} -- {}".format(xi, ga))
    #         plt.pause(.1)
    if args.plot:
        plt.show()
