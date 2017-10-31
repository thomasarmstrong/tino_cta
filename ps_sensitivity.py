#!/usr/bin/env python3
import glob
import numpy as np

# pandas data frames
import pandas as pd

from astropy import units as u

from itertools import chain

from scipy import optimize
from scipy import interpolate

from helper_functions import *

from ctapipe.analysis.sensitivity import (SensitivityPointSource, e_minus_2,
                                          crab_source_rate, cr_background_rate)

from os.path import expandvars
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


# define edges to sort events in
e_bin_edges = np.logspace(-2, np.log10(330), 20)*u.TeV
e_bin_centres = (e_bin_edges[:-1] + e_bin_edges[1:])/2
e_bin_fine_edges = np.logspace(-2, 2.5, 100)*u.TeV
e_bin_fine_centres = (e_bin_fine_edges[:-1] + e_bin_fine_edges[1:])/2

# MC energy ranges:
# gammas: 0.003 to 330 TeV
# proton: 0.004 to 600 TeV
edges_gammas = np.logspace(np.log10(0.003), np.log10(330), 28) * u.TeV
edges_proton = np.logspace(np.log10(0.004), np.log10(600), 30) * u.TeV
edges_electr = np.logspace(np.log10(0.003), np.log10(330), 28) * u.TeV
sensitivity_energy_bin_edges = np.logspace(-2.1, 2.5, 24)*u.TeV
# sensitivity_energy_bin_edges = np.logspace(-2, 2.5, 17)*u.TeV


# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg*u.cm**2*u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# scale MC events to this reference time
observation_time = 50*u.h


channel_map = {'g': "gamma", 'p': "proton", 'e': "electron"}
channel_color_map = {'g': "orange", 'p': "blue", 'e': "red"}
channel_marker_map = {'g': 's', 'p': '^', 'e': 'v'}
channel_linestyle_map = {'g': '-', 'p': '--', 'e': ':'}


def electron_spectrum(e_true_tev):
    """Cosmic-Ray Electron spectrum CTA version, with Fermi Shoulder, in
    units of :math:`\mathrm{TeV^{-1} s^{-1} m^{-2} sr^{-1}}`

    .. math::
       {dN \over dE dA dt d\Omega} =

    """
    e_true_tev /= u.TeV
    number = (6.85e-5 * e_true_tev**-3.21 +
              3.18e-3 / (e_true_tev * 0.776 * np.sqrt(2 * np.pi)) *
              np.exp(-0.5 * (np.log(e_true_tev / 0.107) / 0.776)**2))
    return number * u.Unit("TeV**-1 s**-1 m**-2 sr**-1")


def powerlaw(energy, index, norm, norm_energy=1.0):
    return norm * (energy / norm_energy)**(-index)


def exponential_cutoff(energy, cutoff_energy):
    return np.exp(-energy / cutoff_energy)


def hess_crab_spectrum(e_true_tev, fraction=1.0):
    e_true_tev /= u.TeV
    norm = fraction * u.Quantity(3.76e-11, "cm**-2 s**-1 TeV**-1")
    return powerlaw(e_true_tev, norm=norm,
                    index=2.39, norm_energy=1.0) \
        * exponential_cutoff(e_true_tev, cutoff_energy=14.3)
# crab_source_rate = hess_crab_spectrum


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


def correct_off_angle(data, origin=None):
    import ctapipe.utils.linalg as linalg
    origin = origin or linalg.set_phi_theta(90*u.deg, 20*u.deg)

    reco_dirs = linalg.set_phi_theta(data["phi"]*u.deg.to(u.rad),
                                     data["theta"]*u.deg.to(u.rad)).T
    off_angles = np.arccos(np.clip(np.dot(reco_dirs, origin), -1., 1.))*u.rad
    data["off_angle"] = off_angles.to(u.deg)


def calculate_sensitivities(events, energy_bin_edges, xi_on_scale=1, xi_off_scale=20):
    SensCalc = SensitivityPointSource(
            reco_energies={'g': events['g']['reco_Energy'].values*u.TeV,
                           'p': events['p']['reco_Energy'].values*u.TeV,
                           'e': events['e']['reco_Energy'].values*u.TeV},
            mc_energies={'g': events['g']['MC_Energy'].values*u.TeV,
                         'p': events['p']['MC_Energy'].values*u.TeV,
                         'e': events['e']['MC_Energy'].values*u.TeV},
            flux_unit=flux_unit)

    SensCalc.generate_event_weights(
            n_simulated_events={'g': meta_gammas["n_simulated"],
                                'p': meta_proton["n_simulated"],
                                'e': meta_electr["n_simulated"]},
            generator_areas={'g': np.pi * (meta_gammas["gen_radius"] * u.m)**2,
                             'p': np.pi * (meta_proton["gen_radius"] * u.m)**2,
                             'e': np.pi * (meta_electr["gen_radius"] * u.m)**2},
            observation_time=observation_time,
            spectra={'g': crab_source_rate,
                     'p': cr_background_rate,
                     'e': electron_spectrum},
            e_min_max={'g': (meta_gammas["e_min"], meta_gammas["e_max"])*u.TeV,
                       'p': (meta_proton["e_min"], meta_proton["e_max"])*u.TeV,
                       'e': (meta_electr["e_min"], meta_electr["e_max"])*u.TeV},
            extensions={'p': meta_proton["diff_cone"] * u.deg,
                        'e': meta_electr["diff_cone"] * u.deg},
            generator_gamma={'g': meta_gammas["gen_gamma"],
                             'p': meta_proton["gen_gamma"],
                             'e': meta_electr["gen_gamma"]})

    SensCalc.get_sensitivity(
            alpha=(xi_on_scale/xi_off_scale)**2, n_draws=-1,
            max_background_ratio=.05,
            sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    return SensCalc


def cut_and_sensitivity(cuts, events, energy_bin_edges, xi_on_scale=1, xi_off_scale=20):
    """ throw this into a minimiser """
    ga_cut = cuts[0]
    xi_cut = cuts[1]
    # nt_cut = cuts[2]

    cut_events = {}
    for key in events:
        cut_events[key] = events[key][
            (events[key]["gammaness"] > ga_cut) &
            # (events[key]["NTels_reco"] > nt_cut) &
            (events[key]["off_angle"] < xi_cut)]

    if len(events['g']) < 10 or \
            len(events['g']) < (len(events['p'])+len(events['e'])) * 0.05:
        return 1

    SensCalc = calculate_sensitivities(
            cut_events, energy_bin_edges,
            xi_on_scale=xi_on_scale, xi_off_scale=xi_off_scale)

    if len(SensCalc.sensitivities):
        return SensCalc.sensitivities["Sensitivity"][0]
    else:
        return 1


def get_optimal_splines(events, optimise_bin_edges, k=3):

    cut_events = {}
    cut_energies, ga_cuts, xi_cuts, nt_cuts = [], [], [], []
    for elow, ehigh, emid in zip(optimise_bin_edges[:-1],
                                 optimise_bin_edges[1:],
                                 np.sqrt(optimise_bin_edges[:-1] *
                                         optimise_bin_edges[1:])):

        for key in events:
            cut_events[key] = events[key][
                (events[key]["reco_Energy"] > elow) &
                (events[key]["reco_Energy"] < ehigh)]

        res = optimize.differential_evolution(
                    cut_and_sensitivity,
                    bounds=[(.5, 1), (0, 0.5)],
                    # bounds=[(.5, 1), (0, 0.5), (1, 10)],
                    maxiter=2000, popsize=20,
                    args=(cut_events, np.array([elow/energy_unit,
                                                ehigh/energy_unit])*energy_unit),
            )

        if res.success:
            cut_energies.append(emid.value)
            ga_cuts.append(res.x[0])
            xi_cuts.append(res.x[1])
            # nt_cuts.append(res.x[2])

    spline_ga = interpolate.splrep(cut_energies, ga_cuts, k=k)
    spline_xi = interpolate.splrep(cut_energies, xi_cuts, k=k)
    # spline_nt = interpolate.splrep(cut_energies, nt_cuts, k=k)

    return spline_ga, spline_xi


# ########  ##        #######  ########  ######
# ##     ## ##       ##     ##    ##    ##    ##
# ##     ## ##       ##     ##    ##    ##
# ########  ##       ##     ##    ##     ######
# ##        ##       ##     ##    ##          ##
# ##        ##       ##     ##    ##    ##    ##
# ##        ########  #######     ##     ######

def make_sensitivity_plots(SensCalc, sensitivities,
                           SensCalc_t, sensitivities_t):

    bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.
    bin_centres_p = (edges_proton[1:]+edges_proton[:-1])/2.

    bin_widths_g = np.diff(edges_gammas.value)
    bin_widths_p = np.diff(edges_proton.value)

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
    plt.semilogy(
        sensitivities["Energy"].to(energy_unit),
        (sensitivities["Sensitivity_base"].to(flux_unit) *
         sensitivities["Energy"].to(u.erg)**2),
        color="darkgreen",
        marker="^",
        ls="",
        label="wavelets (no upscale)")

    # tailcuts
    sens_low_t, sens_up_t = (
        (sensitivities_t["Sensitivity"] -
         sensitivities_t["Sensitivity_low"]).to(flux_unit) *
        sensitivities_t["Energy"].to(u.erg)**2,
        (sensitivities_t["Sensitivity_up"] -
         sensitivities_t["Sensitivity"]).to(flux_unit) *
        sensitivities_t["Energy"].to(u.erg)**2)

    plt.errorbar(
        sensitivities_t["Energy"],
        (sensitivities_t["Sensitivity"].to(flux_unit) *
         sensitivities_t["Energy"].to(u.erg)**2).value,
        (sens_low_t.value, sens_up_t.value),
        color="darkorange",
        marker="s", ls="--",
        label="tailcuts")
    plt.semilogy(
        sensitivities_t["Energy"].to(energy_unit),
        (sensitivities_t["Sensitivity_base"].to(flux_unit) *
         sensitivities_t["Energy"].to(u.erg)**2),
        color="darkblue",
        marker="v",
        ls="",
        label="tailcuts (no upscale)")

    plt.legend(title="Obsetvation Time: {}".format(observation_time), loc=1)
    plt.xlabel(r'$E_\mathrm{reco}$' + ' / {:latex}'.format(energy_unit))
    plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
    plt.gca().set_xscale("log")
    plt.grid()
    plt.xlim([1e-2, 2e2])
    plt.ylim([5e-15, 5e-10])

    # plot the sensitivity ratios
    try:
        plt.figure()
        plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[:-2],
                     (sensitivities_t["Sensitivity"].to(flux_unit) *
                      sensitivities_t["Energy"].to(u.erg)**2)[:-2] /
                     (sensitivities["Sensitivity"].to(flux_unit) *
                      sensitivities["Energy"].to(u.erg)**2)[1:-2],
                     label=r"Sens$_\text{tail} / Sens$_\text{wave}$$"
                     )
        plt.legend()
        plt.semilogx(sensitivities_t["Energy"].to(energy_unit)[[0, -1]],
                     [1, 1], ls="--", color="gray")
        plt.xlim(sensitivities_t["Energy"].to(energy_unit)[[0, -1]].value)
        # plt.ylim([.25, 1.1])
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel("ratio")
    except:
        plt.close()

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


def show_gammaness(events, suptitle=None):

    gammas = events['g']
    proton = events['p']
    electr = events['e']

    # 1D gammaness
    e_slices = np.array([0.03, 0.1, 1, np.inf]) * u.TeV
    for e_step in sliding_window(e_slices, step_size=1, window_size=2):
        plt.figure()
        for key in events:
            plt.hist(events[key]["gammaness"][(events[key]["MC_Energy"] > e_step[0]) &
                                              (events[key]["MC_Energy"] < e_step[1])],
                     bins=100, normed=True, alpha=.5,
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], suptitle]),
                     color=channel_color_map[key])
        plt.suptitle(" to ".join([str(e_step[0]), str(e_step[1])]))
        plt.legend(title="integral normalised")
        if args.store:
            save_fig(args.plots_dir +
                     "_".join(["gammaness", str(e_step[0]), str(e_step[1])]))

    gamm_bins = np.linspace(0, 1, 101)
    NTels_bins = np.linspace(0, 50, 21)[:]
    energy_bins = np.linspace(-2, 2.5, 16)[1:]
    energy_bin_centres = (energy_bins[:-1]+energy_bins[1:])/2

    # gammaness vs. number of telescopes

    gamm_vs_ntel_g = np.histogram2d(gammas["NTels_reco"], gammas["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_p = np.histogram2d(proton["NTels_reco"], proton["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_e = np.histogram2d(electr["NTels_reco"], electr["gammaness"],
                                    bins=(NTels_bins, gamm_bins))[0].T
    gamm_vs_ntel_g = gamm_vs_ntel_g / gamm_vs_ntel_g.sum(axis=0)
    gamm_vs_ntel_p = gamm_vs_ntel_p / gamm_vs_ntel_p.sum(axis=0)
    gamm_vs_ntel_e = gamm_vs_ntel_e / gamm_vs_ntel_e.sum(axis=0)

    fig = plt.figure()
    ax1 = plt.subplot(131)
    im = ax1.imshow(np.sqrt(gamm_vs_ntel_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel("NTels")
    ax1.set_ylabel("gammaness")
    ax1.set_title("gammas")

    ax2 = plt.subplot(132, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_ntel_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel("NTels")
    ax2.set_title("protons")

    ax3 = plt.subplot(133, sharey=ax1)
    im = ax3.imshow(np.sqrt(gamm_vs_ntel_e), interpolation='none', origin='lower',
                    aspect='auto', extent=(1, 50, 0, 1), vmin=0, vmax=1.)
    ax3.set_xlabel("NTels")
    ax3.set_title("electrons")

    cb = fig.colorbar(im, ax=[ax1, ax2, ax3], label="sqrt(event fraction per column)")

    if suptitle:
        plt.suptitle(suptitle)

    if args.store:
        save_fig(args.plots_dir + "gammaness_vs_n_tel")

    #
    # gammaness vs. reconstructed energy

    gamm_vs_e_reco_g = np.histogram2d(
            np.log10(gammas["reco_Energy"]), gammas["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_p = np.histogram2d(
            np.log10(proton["reco_Energy"]), proton["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_e = np.histogram2d(
            np.log10(electr["reco_Energy"]), electr["gammaness"],
            bins=(energy_bins, gamm_bins))[0].T
    gamm_vs_e_reco_g = gamm_vs_e_reco_g / gamm_vs_e_reco_g.sum(axis=0)
    gamm_vs_e_reco_p = gamm_vs_e_reco_p / gamm_vs_e_reco_p.sum(axis=0)
    gamm_vs_e_reco_e = gamm_vs_e_reco_e / gamm_vs_e_reco_e.sum(axis=0)

    fig = plt.figure()
    ax1 = plt.subplot(131)
    im = ax1.imshow(np.sqrt(gamm_vs_e_reco_g), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.,
                    cmap=plt.cm.inferno)
    ax1.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax1.set_ylabel("gammaness")
    ax1.set_title("gammas")

    ax2 = plt.subplot(132, sharey=ax1)
    im = ax2.imshow(np.sqrt(gamm_vs_e_reco_p), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.)
    ax2.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax2.set_title("protons")

    ax3 = plt.subplot(133, sharey=ax1)
    im = ax3.imshow(np.sqrt(gamm_vs_e_reco_e), interpolation='none', origin='lower',
                    aspect='auto', extent=(-2, 2.5, 0, 1), vmin=0, vmax=1.)
    ax3.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
    ax3.set_title("electrons")

    cb = fig.colorbar(im, ax=[ax1, ax2, ax3], label="sqrt(event fraction per E-column)")

    if suptitle:
        plt.suptitle(suptitle)

    if args.store:
        save_fig(args.plots_dir + "gammaness_vs_e_reco")

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


def make_performance_plots(events_w, events_t, which=None):

    if not which or "ang_res_verbose" in which:
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

    # angular resolutions

    if not which or ("ang_res" in which or "xi" in which):
        plt.figure()
        for key in events_w:
            xi_68_w = percentiles(events_w[key]["off_angle"],
                                  events_w[key]["reco_Energy"],
                                  e_bin_edges.value, 68)
            xi_68_t = percentiles(events_t[key]["off_angle"],
                                  events_t[key]["reco_Energy"],
                                  e_bin_edges.value, 68)

            plt.plot(e_bin_centres, xi_68_t,
                     color="darkorange",
                     marker=channel_marker_map[key],
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], "tail"]))
            plt.plot(e_bin_centres, xi_68_w,
                     color="darkred",
                     marker=channel_marker_map[key],
                     ls=channel_linestyle_map[key],
                     label="--".join([channel_map[key], "wave"]))
        plt.title("angular resolution")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$\xi_\mathrm{68} / ^\circ$")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.grid()
        plt.legend()

        plt.pause(.1)

    if not which or "gammaness" in which:
        # gammaness plots
        show_gammaness(events_w, "wavelets")
        show_gammaness(events_t, "tailcuts")

    if not which or "energy_migration" in which:
        # MC Energy vs. reco Energy 2D histograms
        for events, mode in zip([events_w, events_t],
                                ["wavelets", "tailcuts"]):
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)
                counts, _, _ = np.histogram2d(events[key]["reco_Energy"],
                                              events[key]["MC_Energy"],
                                              bins=(e_bin_fine_edges, e_bin_fine_edges))
                ax.pcolormesh(e_bin_fine_edges.value, e_bin_fine_edges.value, counts.T)
                plt.plot(e_bin_fine_edges.value[[0, -1]], e_bin_fine_edges.value[[0, -1]],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$E_\mathrm{MC}$ / TeV")
                ax.set_xscale("log")
                ax.set_yscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)

            plt.pause(.1)

    if not which or "DeltaE" in which:
        # (reco Energy - MC Energy) 2D histograms
        for events, mode in zip([events_w, events_t],
                                ["wavelets", "tailcuts"]):

            # (reco Energy - MC Energy) vs. reco Energy 2D histograms
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)

                counts, _, _ = np.histogram2d(
                            events[key]["reco_Energy"],
                            events[key]["reco_Energy"] - events[key]["MC_Energy"],
                            bins=(e_bin_fine_edges, np.linspace(-1, 1, 100)))
                ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-1, 1, 100),
                              np.sqrt(counts.T))
                plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{reco}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC})$ / TeV")
                ax.set_xscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)
            plt.pause(.1)

            # (reco Energy - MC Energy) vs. MC Energy 2D histograms
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)

                counts, _, _ = np.histogram2d(
                            events[key]["MC_Energy"],
                            events[key]["reco_Energy"] - events[key]["MC_Energy"],
                            bins=(e_bin_fine_edges, np.linspace(-1, 1, 100)))
                ax.pcolormesh(e_bin_fine_edges.value, np.linspace(-1, 1, 100),
                              np.sqrt(counts.T))
                plt.plot(e_bin_fine_edges.value[[0, -1]], [0, 0],
                         color="darkgreen")
                plt.title(channel_map[key])
                ax.set_xlabel(r"$E_\mathrm{MC}$ / TeV")
                if i == 0:
                    ax.set_ylabel(r"$(E_\mathrm{reco} - E_\mathrm{MC})$ / TeV")
                ax.set_xscale("log")
                plt.grid()

            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)
            plt.pause(.1)

    if not which or "Energy_resolution" in which:
        # energy resolution as 68th percentile of the relative reconstructed error binned
        # in reconstructed energy
        rel_DeltaE_w = np.abs(events_w['g']["reco_Energy"] -
                              events_w['g']["MC_Energy"])/events_w['g']["reco_Energy"]
        DeltaE68_w_ebinned = percentiles(rel_DeltaE_w, events_w['g']["reco_Energy"],
                                         e_bin_edges.value, 68)

        rel_DeltaE_t = np.abs(events_t['g']["reco_Energy"] -
                              events_t['g']["MC_Energy"])/events_t['g']["reco_Energy"]
        DeltaE68_t_ebinned = percentiles(rel_DeltaE_t, events_t['g']["reco_Energy"],
                                         e_bin_edges.value, 68)

        plt.figure()
        plt.plot(e_bin_centres, DeltaE68_t_ebinned, label="gamma -- tail",
                 marker='v', color="darkorange")
        plt.plot(e_bin_centres, DeltaE68_w_ebinned, label="gamma -- wave",
                 marker='^', color="darkred")
        plt.title("Energy Resolution")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{reco}$")
        plt.gca().set_xscale("log")
        plt.grid()
        plt.legend()

        plt.pause(.1)

        # energy resolution binned in MC Energy
        for key in ['g', 'e']:
            plt.figure()
            for events, mode in zip([events_w, events_t],
                                    ["wavelets", "tailcuts"]):

                rel_DeltaE = np.abs(events[key]["reco_Energy"] -
                                    events[key]["MC_Energy"])/events[key]["MC_Energy"]
                DeltaE68_ebinned = percentiles(rel_DeltaE, events[key]["MC_Energy"],
                                               e_bin_edges.value, 68)

                plt.plot(e_bin_centres, DeltaE68_ebinned,
                         label=" -- ".join([channel_map[key], mode]),
                         marker=channel_marker_map[key],
                         color="darkred" if "wave" in mode else "darkorange")
            plt.title("Energy Resolution")
            plt.xlabel(r"$E_\mathrm{MC}$ / TeV")
            plt.ylabel(r"$(|E_\mathrm{reco} - E_\mathrm{MC}|)_{68}/E_\mathrm{MC}$")
            plt.gca().set_xscale("log")
            plt.grid()
            plt.legend()

            plt.pause(.1)

    if not which or "Energy_bias" in which:
        # Ebias as median of 1-E_reco/E_MC
        for key in ['g', 'e']:
            plt.figure()
            for events, mode in zip([events_w, events_t],
                                    ["wavelets", "tailcuts"]):
                Ebias = 1 - (events[key]["reco_Energy"]/events[key]["MC_Energy"])
                Ebias_medians = percentiles(Ebias, events[key]["reco_Energy"],
                                            e_bin_edges.value, 50)
                plt.plot(e_bin_centres, Ebias_medians,
                         label=" -- ".join([channel_map[key], mode]),
                         marker=channel_marker_map[key],
                         color="darkred" if "wave" in mode else "darkorange")
            plt.title("Energy Bias")
            plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
            plt.ylabel(r"$(1 - E_\mathrm{reco}/E_\mathrm{MC})_{50}$")
            plt.ylim([-0.2, .3])
            plt.gca().set_xscale("log")
            plt.legend()
            plt.grid()

            plt.pause(.1)

    if not which or any(what in which for what in ["gen_spectrum", "expected_events",
                                                "effective_areas"]):
        bin_centres, bin_widths = {}, {}
        bin_centres['g'] = (edges_gammas[:-1] + edges_gammas[1:])/2
        bin_centres['p'] = (edges_proton[:-1] + edges_proton[1:])/2
        bin_centres['e'] = (edges_electr[:-1] + edges_electr[1:])/2
        bin_widths['g'] = np.diff(edges_gammas)
        bin_widths['p'] = np.diff(edges_proton)
        bin_widths['e'] = np.diff(edges_electr)

        for events, mode in zip([events_w], ["wavelets"]):
            SensCalc = SensitivityPointSource(
                    reco_energies={'g': events['g']['reco_Energy'].values*u.TeV,
                                   'p': events['p']['reco_Energy'].values*u.TeV,
                                   'e': events['e']['reco_Energy'].values*u.TeV},
                    mc_energies={'g': events['g']['MC_Energy'].values*u.TeV,
                                 'p': events['p']['MC_Energy'].values*u.TeV,
                                 'e': events['e']['MC_Energy'].values*u.TeV},
                    energy_bin_edges={'g': edges_gammas,
                                      'p': edges_proton,
                                      'e': edges_electr},
                    flux_unit=flux_unit)

            SensCalc.generate_event_weights(
                    n_simulated_events={'g': meta_gammas["n_simulated"],
                                        'p': meta_proton["n_simulated"],
                                        'e': meta_electr["n_simulated"]},
                    generator_areas={'g': np.pi *
                                     (meta_gammas["gen_radius"] * u.m)**2,
                                     'p': np.pi *
                                     (meta_proton["gen_radius"] * u.m)**2,
                                     'e': np.pi *
                                     (meta_electr["gen_radius"] * u.m)**2},
                    observation_time=observation_time,
                    spectra={'g': crab_source_rate,
                             'p': cr_background_rate,
                             'e': electron_spectrum},
                    e_min_max={'g': (meta_gammas["e_min"],
                                     meta_gammas["e_max"])*u.TeV,
                               'p': (meta_proton["e_min"],
                                     meta_proton["e_max"])*u.TeV,
                               'e': (meta_electr["e_min"],
                                     meta_electr["e_max"])*u.TeV},
                    extensions={'p': meta_proton["diff_cone"] * u.deg,
                                'e': meta_electr["diff_cone"] * u.deg},
                    generator_gamma={'g': meta_gammas["gen_gamma"],
                                     'p': meta_proton["gen_gamma"],
                                     'e': meta_electr["gen_gamma"]})
            SensCalc.get_effective_areas(
                    generator_areas={'g': np.pi *
                                     (meta_gammas["gen_radius"] * u.m)**2,
                                     'p': np.pi *
                                     (meta_proton["gen_radius"] * u.m)**2,
                                     'e': np.pi *
                                     (meta_electr["gen_radius"] * u.m)**2},
                    n_simulated_events={'g': meta_gammas["n_simulated"],
                                        'p': meta_proton["n_simulated"],
                                        'e': meta_electr["n_simulated"]},
                    generator_spectra={'g': e_minus_2,
                                       'p': e_minus_2,
                                       'e': e_minus_2},
                    )
            SensCalc.get_expected_events()

        if not which or "gen_spectrum" in which:
            # plot MC generator spectrum and selected spectrum
            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
            for i, key in enumerate(events):
                ax = axs[i]
                plt.sca(ax)
                plt.bar(bin_centres[key].value,
                        SensCalc.generator_energy_hists[key], label="generated",
                        align="center", width=bin_widths[key].value)
                plt.bar(bin_centres[key].value,
                        SensCalc.selected_events[key], label="selected",
                        align="center", width=bin_widths[key].value)
                plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres[key].unit)+"}$")
                if i == 0:
                    plt.ylabel("number of (unweighted) events")
                plt.gca().set_xscale("log")
                plt.gca().set_yscale("log")
                plt.title(channel_map[key])
                plt.legend()
            plt.suptitle(mode)
            plt.subplots_adjust(left=.1, wspace=.1)
            plt.pause(.1)

        if not which or "expected_events" in which:
            # plot the number of expected events in each energy bin
            plt.figure()
            for key in ['p', 'e', 'g']:
                plt.bar(
                    bin_centres[key].value,
                    SensCalc.exp_events_per_energy_bin[key],
                    label=channel_map[key],
                    color=channel_color_map[key],
                    align="center", width=bin_widths[key].value, alpha=.75)
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")

            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres['g'].unit)+"}$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.legend()
            plt.pause(.1)

        if not which or "effective_areas" in which:
            # plot effective area
            plt.figure()  # figsize=(16, 8))
            plt.suptitle("Effective Areas")
            for key in ['p', 'e', 'g']:
                plt.loglog(
                    bin_centres[key],
                    SensCalc.effective_areas[key],
                    label=channel_map[key], color=channel_color_map[key],
                    marker=channel_marker_map[key])
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres['g'].unit)+"}$")
            plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
            plt.title(mode)
            plt.legend()
            plt.pause(.1)


if __name__ == "__main__":
    np.random.seed(19)

    parser = make_argparser()
    parser.add_argument('--infile', type=str, default="classified_events")
    args = parser.parse_args()

    # load meta data from disk
    meta_data_file = "{}/meta_data.yml".format(args.indir)
    meta_data = yaml.load(open(meta_data_file), Loader=Loader)
    meta_units = meta_data["units"]
    meta_gammas = meta_data["gamma"]
    meta_proton = meta_data["proton"]
    meta_electr = meta_data["electron"]

    meta_gammas["n_simulated"] = meta_gammas["n_files"] * meta_gammas["n_events_per_file"]
    meta_proton["n_simulated"] = meta_proton["n_files"] * meta_proton["n_events_per_file"]
    meta_electr["n_simulated"] = meta_electr["n_files"] * meta_electr["n_events_per_file"]

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

    events_w = {'g': gammas_w_o, 'p': proton_w_o, 'e': electr_w_o}

    gammas_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "gamma", "tail"), "reco_events")
    proton_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "proton", "tail"), "reco_events")
    electr_t_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "electron", "tail"), "reco_events")

    events_t = {'g': gammas_t_o, 'p': proton_t_o, 'e': electr_t_o}

    make_performance_plots(events_w, events_t, which=["gen_spectrum", "expected_events",
                                                      ])

    plt.show()

    # ##      ##    ###    ##     ## ########
    # ##  ##  ##   ## ##   ##     ## ##
    # ##  ##  ##  ##   ##  ##     ## ##
    # ##  ##  ## ##     ## ##     ## ######
    # ##  ##  ## #########  ##   ##  ##
    # ##  ##  ## ##     ##   ## ##   ##
    #  ###  ###  ##     ##    ###    ########

    spline_w_ga, spline_w_xi = \
        get_optimal_splines(events_w, sensitivity_energy_bin_edges[::2], k=2)

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(121)
    plt.plot(cut_energies, ga_cuts, label="crit. values", ls="", marker="^")
    plt.plot(e_bin_fine_edges, interpolate.splev(e_bin_fine_edges, spline_t_ga),
             label="spline fit")

    plt.xlabel("Energy / TeV")
    plt.ylabel("gammaness")
    plt.gca().set_xscale("log")
    plt.legend()

    fig.add_subplot(122)
    plt.plot(cut_energies, xi_cuts, label="crit. values", ls="", marker="^")
    plt.plot(e_bin_fine_edges, interpolate.splev(e_bin_fine_edges, spline_t_xi),
             label="spline fit")
    plt.xlabel("Energy / TeV")
    plt.ylabel("xi / degree")
    plt.gca().set_xscale("log")
    plt.legend()

    # fig.add_subplot(133)
    # plt.plot(cut_energies, nt_cuts, label="crit. values", ls="", marker="^")
    # plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_t_nt),
    #          label="spline fit")
    # plt.xlabel("Energy / TeV")
    # plt.ylabel("number telescopes")
    # plt.gca().set_xscale("log")
    # plt.legend()

    plt.pause(.1)

    cut_events_w = {}
    for key in events_w:
        cut_events_w[key] = events_w[key][
            (events_w[key]["gammaness"] >
             interpolate.splev(events_w[key]["reco_Energy"], spline_w_ga)) &
            # (events_w[key]["NTels_reco"] >
            #  interpolate.splev(events_w[key]["reco_Energy"], spline_w_nt)) &
            (events_w[key]["off_angle"] <
             interpolate.splev(events_w[key]["reco_Energy"], spline_w_xi))]

    sens_w = calculate_sensitivities(cut_events_w, sensitivity_energy_bin_edges)

    # ########    ###    #### ##
    #    ##      ## ##    ##  ##
    #    ##     ##   ##   ##  ##
    #    ##    ##     ##  ##  ##
    #    ##    #########  ##  ##
    #    ##    ##     ##  ##  ##
    #    ##    ##     ## #### ########

    spline_t_ga, spline_t_xi = \
        get_optimal_splines(events_t, sensitivity_energy_bin_edges[::2], k=2)

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(121)
    plt.plot(cut_energies, ga_cuts, label="crit. values", ls="", marker="^")
    plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_t_ga),
             label="spline fit")

    plt.xlabel("Energy / TeV")
    plt.ylabel("gammaness")
    plt.gca().set_xscale("log")
    plt.legend()

    fig.add_subplot(122)
    plt.plot(cut_energies, xi_cuts, label="crit. values", ls="", marker="^")
    plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_t_xi),
             label="spline fit")
    plt.xlabel("Energy / TeV")
    plt.ylabel("xi / degree")
    plt.gca().set_xscale("log")
    plt.legend()

    # fig.add_subplot(133)
    # plt.plot(cut_energies, nt_cuts, label="crit. values", ls="", marker="^")
    # plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_t_nt),
    #          label="spline fit")
    # plt.xlabel("Energy / TeV")
    # plt.ylabel("number telescopes")
    # plt.gca().set_xscale("log")
    # plt.legend()

    plt.pause(.1)

    cut_events_t = {}
    for key in events_t:
        cut_events_t[key] = events_t[key][
            (events_t[key]["gammaness"] >
             interpolate.splev(events_t[key]["reco_Energy"], spline_t_ga)) &
            # (events_t[key]["NTels_reco"] >
            #  interpolate.splev(events_t[key]["reco_Energy"], spline_t_nt)) &
            (events_t[key]["off_angle"] <
             interpolate.splev(events_t[key]["reco_Energy"], spline_t_xi))]

    sens_t = calculate_sensitivities(events_t, sensitivity_energy_bin_edges)

    make_sensitivity_plots(sens_w, sens_w.sensitivities,
                           sens_t, sens_t.sensitivities)

    # make_performance_plots(sens_w, gammas_w_rcut, proton_w_rcut,
    #                        sens_t, gammas_t_rcut, proton_t_rcut)

    # make_performance_plots(sens_w, gammas_w, proton_w,
    #                        sens_t, gammas_t, proton_t)

    if args.plot:
        plt.show()
