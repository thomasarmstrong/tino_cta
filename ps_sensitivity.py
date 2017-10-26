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


def find_optimal_cuts(cuts, events, energy_bin_edges, xi_on_scale=1, xi_off_scale=20):
    """ throw this into a minimiser """
    ga_cut = cuts[0]
    # xi_cut = 10**cuts[1]

    cut_events = {}
    for key in events:
        cut_events[key] = events[key][
            (events[key]["gammaness"] > ga_cut)]
            # (events[key]["off_angle"] < xi_cut)]

    if len(events['g']) < 10:
        return 1

    SensCalc = calculate_sensitivities(
            cut_events, energy_bin_edges,
            xi_on_scale=xi_on_scale, xi_off_scale=xi_off_scale)

    if len(SensCalc.sensitivities):
        return SensCalc.sensitivities["Sensitivity"][0]
    else:
        return 1


def main(xi_percentile={'w': 68, 't': 68}, xi_on_scale=1, xi_off_scale=20,
         ga_percentile={'w': 99, 't': 99}):

    print()
    print("gammas simulated:", meta_gammas["n_simulated"])
    print("proton simulated:", meta_proton["n_simulated"])
    print("electr simulated:", meta_electr["n_simulated"])
    print()
    print("observation time:", observation_time)

    print("\n")
    print("gammas present (wavelets):", len(gammas_o))
    print("proton present (wavelets):", len(proton_o))
    print("electr present (wavelets):", len(electr_o))
    # print()
    # print("gammas present (tailcuts):", len(gammas_t_o))
    # print("proton present (tailcuts):", len(proton_t_o))
    # print("electr present (tailcuts):", len(electr_t_o))

    #  ######      ###    ##     ## ##     ##    ###    ##    ##  ######
    # ##    ##    ## ##   ###   ### ###   ###   ## ##   ###   ## ##    ##
    # ##         ##   ##  #### #### #### ####  ##   ##  ####  ## ##
    # ##   #### ##     ## ## ### ## ## ### ## ##     ## ## ## ##  ######
    # ##    ##  ######### ##     ## ##     ## ######### ##  ####       ##
    # ##    ##  ##     ## ##     ## ##     ## ##     ## ##   ### ##    ##
    #  ######   ##     ## ##     ## ##     ## ##     ## ##    ##  ######

    g_cuts_w = percentiles(
            proton_o["gammaness"], proton_o["reco_Energy"],
            e_bin_edges.value, ga_percentile['w'])

    spline_g = interpolate.splrep((e_bin_edges[1:]+e_bin_edges[:-1]).value / 2, g_cuts_w)

    # perform gammaness selection
    gammas_g = gammas_o[
            gammas_o["gammaness"] >
            interpolate.splev(gammas_o["reco_Energy"], spline_g)]
    proton_g = proton_o[
            proton_o["gammaness"] >
            interpolate.splev(proton_o["reco_Energy"], spline_g)]
    electr_g = electr_o[
            electr_o["gammaness"] >
            interpolate.splev(electr_o["reco_Energy"], spline_g)]

    if False:
        plt.figure()
        plt.semilogx(e_bin_centres, g_cuts_w, ls="", marker="^",
                     label="crit. values -- {} \%".format(ga_percentile['w']))
        plt.semilogx(e_bin_fine_centres,
                     interpolate.splev(e_bin_fine_centres.value, spline_g),
                     ls="-", marker="", label="spline fit")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel("gammaness")
        plt.legend()
        plt.pause(.1)

        plt.figure()
        plt.hist2d(np.log10(gammas_g["reco_Energy"]), gammas_g["gammaness"],
                   bins=50)
        plt.plot(np.log10(e_bin_fine_centres.value),
                 interpolate.splev(e_bin_fine_centres.value, spline_g),
                 ls="-", marker="", label="spline fit")

    # ##     ## ####     ######  ##     ## ########
    #  ##   ##   ##     ##    ## ##     ##    ##
    #   ## ##    ##     ##       ##     ##    ##
    #    ###     ##     ##       ##     ##    ##
    #   ## ##    ##     ##       ##     ##    ##
    #  ##   ##   ##     ##    ## ##     ##    ##
    # ##     ## ####     ######   #######     ##

    xi_cuts = percentiles(gammas_g["off_angle"], gammas_g["reco_Energy"],
                          e_bin_edges.value, xi_percentile['w'])

    # fit a spline to the cut values in xi
    spline_x = interpolate.splrep((e_bin_edges[1:]+e_bin_edges[:-1]).value / 2, xi_cuts)

    if True:
        plt.figure()
        plt.loglog(e_bin_centres, xi_cuts,
                   color="darkred", marker="^", ls="",
                   label="MC wave -- {} %".format(xi_percentile['w']))
        plt.loglog(e_bin_fine_centres,
                   interpolate.splev(e_bin_fine_centres.value, spline_x),
                   color="darkred", label="spline fit")
        plt.title("on-region size")
        plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
        plt.ylabel(r"$\xi_\mathrm{cut} / ^\circ$")
        plt.grid()
        plt.legend()
        plt.pause(.1)

    gammas_rcut = gammas_g[
            gammas_g["off_angle"] <
            interpolate.splev(gammas_g["reco_Energy"], spline_x)*xi_on_scale]
    proton_rcut = proton_g[
            proton_g["off_angle"] <
            interpolate.splev(proton_g["reco_Energy"], spline_x)*xi_off_scale]
    electr_rcut = electr_g[
            electr_g["off_angle"] <
            interpolate.splev(electr_g["reco_Energy"], spline_x)*xi_off_scale]

    print("\n")
    print("gammas selected (wavelets):", len(gammas_rcut))
    print("proton selected (wavelets):", len(proton_rcut))
    print("electr selected (wavelets):", len(electr_rcut))

    SensCalc = SensitivityPointSource(
            reco_energies={'g': gammas_rcut['reco_Energy'].values*u.TeV,
                           'p': proton_rcut['reco_Energy'].values*u.TeV,
                           'e': electr_rcut['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas_rcut['MC_Energy'].values*u.TeV,
                         'p': proton_rcut['MC_Energy'].values*u.TeV,
                         'e': electr_rcut['MC_Energy'].values*u.TeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton,
                              'e': edges_gammas},
            flux_unit=flux_unit)

    SensCalc.get_effective_areas(
            n_simulated_events={'g': meta_gammas["n_simulated"],
                                'p': meta_proton["n_simulated"],
                                'e': meta_electr["n_simulated"]},
            generator_spectra={'g': e_minus_2,
                               'p': e_minus_2,
                               'e': e_minus_2},
            generator_areas={'g': np.pi * (meta_gammas["gen_radius"] * u.m)**2,
                             'p': np.pi * (meta_proton["gen_radius"] * u.m)**2,
                             'e': np.pi * (meta_electr["gen_radius"] * u.m)**2},
    )

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

    print("\n")
    print("gammas expected (wavelets):", np.sum(SensCalc.event_weights["g"]))
    print("proton expected (wavelets):", np.sum(SensCalc.event_weights["p"]))
    print("electr expected (wavelets):", np.sum(SensCalc.event_weights["e"]))

    return SensCalc, gammas_rcut, proton_rcut, electr_rcut

    # make_performance_plots(gammas_w_o, proton_w_o, gammas_t_o, proton_t_o)
    # make_performance_plots(gammas_w_g, proton_w_g, gammas_t_g, proton_t_g)
    # make_performance_plots(gammas_w_rcut, proton_w_rcut)

    # show_gammaness(gammas_w_rcut, proton_w_rcut, "wave")
    # show_gammaness(gammas_t_rcut, proton_t_rcut, "tail")


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


def make_performance_plots(SensCalc_w, gammas_w, proton_w,
                           SensCalc_t, gammas_t, proton_t):

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
    plt.plot(e_bin_centres, xi_68_gt,
             color="darkorange", marker="v", ls="-",
             label="gamma -- tail")
    # plt.plot(e_bin_centres, xi_68_pt,
    #              color="darkorange", marker="o", ls=":",
    #              label="proton -- tail")
    plt.plot(e_bin_centres, xi_68_gw,
             color="darkred", marker="^", ls="-",
             label="gamma -- wave")
    # plt.plot(e_bin_centres, xi_68_pw,
    #              color="darkred", marker="o", ls=":",
    #              label="proton -- wave")
    plt.title("angular resolution")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$\xi_\mathrm{68} / ^\circ$")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.legend()

    plt.pause(.1)

    # MC Energy vs. reco Energy 2D histograms
    fig = plt.figure()
    ax = plt.subplot(121)
    counts_g, _, _ = np.histogram2d(gammas_w["MC_Energy"],
                                    gammas_w["reco_Energy"],
                                    bins=(e_bin_fine_edges, e_bin_fine_edges))
    ax.pcolormesh(e_bin_fine_edges.value, e_bin_fine_edges.value, counts_g)
    plt.plot(e_bin_fine_edges.value[[0, -1]], e_bin_fine_edges.value[[0, -1]],
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
                                    bins=(e_bin_fine_edges, e_bin_fine_edges))
    ax.pcolormesh(e_bin_fine_edges.value, e_bin_fine_edges.value, counts_p)
    plt.plot(e_bin_fine_edges.value[[0, -1]], e_bin_fine_edges.value[[0, -1]],
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

    # Ebias as median of 1-E_reco/E_MC
    Ebias_w = 1 - (gammas_w["reco_Energy"]/gammas_w["MC_Energy"])
    Ebias_w_medians = percentiles(Ebias_w, gammas_w["reco_Energy"],
                                  e_bin_edges.value, 50)
    Ebias_t = 1 - (gammas_t["reco_Energy"]/gammas_t["MC_Energy"])
    Ebias_t_medians = percentiles(Ebias_t, gammas_t["reco_Energy"],
                                  e_bin_edges.value, 50)

    plt.figure()
    plt.plot(e_bin_centres, Ebias_t_medians, label="gamma -- tail",
             marker='v', color="darkorange")
    plt.plot(e_bin_centres, Ebias_w_medians, label="gamma -- wave",
             marker='^', color="darkred")
    plt.title("Energy Bias")
    plt.xlabel(r"$E_\mathrm{reco}$ / TeV")
    plt.ylabel(r"$(1 - E_\mathrm{reco}/E_\mathrm{MC})_{50}$")
    plt.ylim([-0.2, .3])
    plt.gca().set_xscale("log")
    plt.legend()
    plt.grid()


    if args.verbose or True:
        bin_centres_g = (edges_gammas[:-1] + edges_gammas[1:])/2
        bin_centres_p = (edges_proton[:-1] + edges_proton[1:])/2

        bin_widths_g = (edges_gammas[1:] - edges_gammas[:-1])
        bin_widths_p = (edges_proton[1:] - edges_proton[:-1])

        # plot MC generator spectrum and selected spectrum
        plt.figure()
        plt.subplot(121)
        plt.bar(bin_centres_g.value,
                SensCalc_w.generator_energy_hists['g'], label="generated",
                align="center", width=bin_widths_g.value)
        plt.bar(bin_centres_g.value,
                SensCalc_w.selected_events['g'], label="selected",
                align="center", width=bin_widths_g.value)
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        plt.ylabel("number of events")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title("gammas -- wavelets")
        plt.legend()

        plt.subplot(122)
        plt.bar(bin_centres_p.value,
                SensCalc_w.generator_energy_hists['p'], label="generated",
                align="center", width=bin_widths_p.value)
        plt.bar(bin_centres_p.value,
                SensCalc_w.selected_events['p'], label="selected",
                align="center", width=bin_widths_p.value)
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        plt.ylabel("number of events")
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title("protons -- wavelets")
        plt.legend()

        # plot the number of expected events in each energy bin
        plt.figure()
        plt.bar(
                bin_centres_p.value,
                SensCalc_w.exp_events_per_energy_bin['p'], label="proton",
                align="center", width=np.diff(edges_proton.value), alpha=.75)
        plt.bar(
                bin_centres_g.value,
                SensCalc_w.exp_events_per_energy_bin['g'], label="gamma",
                align="center", width=np.diff(edges_gammas.value), alpha=.75)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")

        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
        plt.ylabel("expected events in {}".format(observation_time))
        plt.legend()

        # plot effective area
        kwargs = {"n_simulated_events": {'g': meta_gammas["n_simulated"],
                                         'p': meta_proton["n_simulated"],
                                         'e': meta_electr["n_simulated"]},
                  "generator_spectra": {'g': e_minus_2,
                                        'p': e_minus_2,
                                        'e': e_minus_2},
                  "generator_areas": {'g': np.pi * (meta_gammas["gen_radius"] * u.m)**2,
                                      'p': np.pi * (meta_proton["gen_radius"] * u.m)**2,
                                      'e': np.pi * (meta_electr["gen_radius"] * u.m)**2}}
        SensCalc_w.energy_bin_edges = {'g': edges_gammas,
                                       'p': edges_proton,
                                       'e': edges_electr},
        SensCalc_t.energy_bin_edges = {'g': edges_gammas,
                                       'p': edges_proton,
                                       'e': edges_electr},
        SensCalc_w.get_effective_areas(**kwargs)
        SensCalc_t.get_effective_areas(**kwargs)
        plt.figure()  # figsize=(16, 8))
        plt.suptitle("Effective Areas")
        # plt.subplot(121)
        plt.loglog(
            bin_centres_g,
            SensCalc_w.effective_areas['g'],
            label="wavelets", color="darkred", marker="^")
        plt.loglog(
            bin_centres_g,
            SensCalc_t.effective_areas['g'],
            label="tailcuts", color="darkorange", marker="v")
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
    gammas_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "gamma", args.mode), "reco_events")
    proton_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "proton", args.mode), "reco_events")
    electr_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "electron", args.mode), "reco_events")

    if args.mode == "wave":
        # FUCK FUCK FUCK FUCK
        correct_off_angle(gammas_o)
        correct_off_angle(proton_o)
        correct_off_angle(electr_o)

    # test_ga = 1-np.logspace(-2, -0.5, 15)
    # test_xi = np.logspace(-3, -0.5, 15)
    cut_energies, ga_cuts, xi_cuts = [], [], []
    for elow, ehigh, emid in zip(sensitivity_energy_bin_edges[:-1],
                                 sensitivity_energy_bin_edges[1:],
                                 np.sqrt(sensitivity_energy_bin_edges[:-1] *
                                         sensitivity_energy_bin_edges[1:])):
        print()
        print(emid)

        events = {}
        events['g'] = gammas_o[(gammas_o["reco_Energy"] > elow) &
                               (gammas_o["reco_Energy"] < ehigh)]
        events['p'] = proton_o[(proton_o["reco_Energy"] > elow) &
                               (proton_o["reco_Energy"] < ehigh)]
        events['e'] = electr_o[(electr_o["reco_Energy"] > elow) &
                               (electr_o["reco_Energy"] < ehigh)]

        # min_sens = 1
        # for ga in test_ga:
        #     for xi in test_xi:
        #         sens = find_optimal_cuts(
        #             [ga, xi], events, np.array([elow/energy_unit,
        #                                         ehigh/energy_unit])*energy_unit)
        #         if sens < min_sens:
        #             min_sens = sens
        #             min_ga = ga
        #             min_xi = xi
        #
        # if min_sens < 1:
        #     ga_cuts.append(min_ga)
        #     xi_cuts.append(min_xi)
        #     cut_energies.append(emid.value)

        res = optimize.differential_evolution(
                    find_optimal_cuts,
                    bounds=[(.5, 1)],
                    # bounds=[(-5, -0.1)],
                    # bounds=[(-5, -0.1), (-5, 0)],
                    maxiter=2000, popsize=20,
                    args=(events, np.array([elow/energy_unit,
                                            ehigh/energy_unit])*energy_unit),
            )

        if res.success:
            cut_energies.append(emid.value)
            ga_cuts.append(1-10**res.x[0])
            # xi_cuts.append(10**res.x[1])
            # print(ga_cuts[-1], xi_cuts[-1])

    spline_ga = interpolate.splrep(cut_energies, ga_cuts)
    # spline_xi = interpolate.splrep(cut_energies, xi_cuts)
    spline_test_points = np.logspace(*(np.log10(cut_energies)[[0, -1]]), 100)
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(121)
    plt.plot(cut_energies, ga_cuts, label="crit. values", ls="", marker="^")
    plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_ga),
             label="spline fit")

    plt.xlabel("Energy / TeV")
    plt.ylabel("gammaness")
    plt.gca().set_xscale("log")
    plt.legend()

    # fig.add_subplot(122)
    # plt.plot(cut_energies, xi_cuts, label="crit. values", ls="", marker="^")
    # plt.plot(spline_test_points, interpolate.splev(spline_test_points, spline_xi),
    #          label="spline fit")
    # plt.xlabel("Energy / TeV")
    # plt.ylabel("xi / degree")
    # plt.gca().set_xscale("log")
    # plt.legend()

    plt.pause(.1)
    plt.show()

    events = {}
    events['g'] = gammas_o[
            (gammas_o["gammaness"] >
             interpolate.splev(gammas_o["reco_Energy"], spline_ga)) &
            (gammas_o["off_angle"] >
             interpolate.splev(gammas_o["reco_Energy"], spline_xi))]
    events['p'] = proton_o[
            (proton_o["gammaness"] >
             interpolate.splev(proton_o["reco_Energy"], spline_ga)) &
            (proton_o["off_angle"] >
             interpolate.splev(proton_o["reco_Energy"], spline_xi))]
    events['e'] = electr_o[
            (electr_o["gammaness"] >
             interpolate.splev(electr_o["reco_Energy"], spline_ga)) &
            (electr_o["off_angle"] >
             interpolate.splev(electr_o["reco_Energy"], spline_xi))]

    sens_w = calculate_sensitivities(events, sensitivity_energy_bin_edges)
    make_sensitivity_plots(sens_w, sens_w.sensitivities,
                           sens_w, sens_w.sensitivities)

    plt.show()

    gammas_w, proton_w, electr_w = gammas_o, proton_o, electr_o
    sens_w, gammas_w_rcut, proton_w_rcut, electr_w_rcut = main(
                xi_percentile={'w': 68, 't': 68},
                ga_percentile={'w': 99.99, 't': 99.99})

    gammas_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "gamma", "tail"), "reco_events")
    proton_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "proton", "tail"), "reco_events")
    electr_o = pd.read_hdf("{}/{}_{}_{}.h5".format(
            args.indir, args.infile, "electron", "tail"), "reco_events")

    gammas_t, proton_t, electr_t = gammas_o, proton_o, electr_o
    sens_t, gammas_t_rcut, proton_t_rcut, electr_t_rcut = main(
                xi_percentile={'w': 68, 't': 68},
                ga_percentile={'w': 99.99, 't': 99.99})

    make_sensitivity_plots(sens_w, sens_w.sensitivities,
                           sens_t, sens_t.sensitivities)

    # make_performance_plots(sens_w, gammas_w_rcut, proton_w_rcut,
    #                        sens_t, gammas_t_rcut, proton_t_rcut)

    # make_performance_plots(sens_w, gammas_w, proton_w,
    #                        sens_t, gammas_t, proton_t)

    if args.plot:
        plt.show()
