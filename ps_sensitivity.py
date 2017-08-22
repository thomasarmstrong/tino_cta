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

from helper_functions import *

from ctapipe.analysis.sensitivity import (SensitivityPointSource, e_minus_2,
                                          crab_source_rate, cr_background_rate)

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
# plt.style.use('t_slides')


# MC energy ranges:
# gammas: 0.1 to 330 TeV
# proton: 0.1 to 600 TeV
edges_gammas = np.logspace(2, np.log10(330000), 28) * u.GeV
edges_proton = np.logspace(2, np.log10(600000), 30) * u.GeV

sensitivity_energy_bin_edges = np.logspace(-1, 3, 17)*u.TeV


# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg*u.cm**2*u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# scale MC events to this reference time
observation_time = 50*u.h


def open_pytable_as_pandas(filename, mode='r'):
    pyt_infile = tb.open_file(filename, mode=mode)
    pyt_table = pyt_infile.root.reco_events

    return pd.DataFrame(pyt_table[:])


def fitfunc(x, a, b, c, d, e=0, f=0, g=0, h=0):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7


def fitfunc_log(x, a, b, c, d, e=0, f=0, g=0, h=0):
    x = np.log10(x)
    return fitfunc(x, a, b, c, d, e, f, g, h)


def main_const_theta_cut():

    def selection_mask(event_table, ntels=3, gammaness=.75, r_max=0.1*u.deg):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness) &
                (event_table["off_angle"] < r_max))

    apply_cuts = True
    gammaness_wave = .75
    gammaness_tail = .75
    r_max_gamm_wave = 0.05*u.deg
    r_max_gamm_tail = 0.05*u.deg
    r_max_prot = 3*u.deg

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * (498-14)
    NProton_simulated = NProton_per_File * (6998-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "gamma", "wave"))

    proton = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "proton", "wave"))

    print()
    print("gammas present (wavelets):", len(gammas))
    print("proton present (wavelets):", len(proton))

    # applying some cuts
    if apply_cuts:
        gammas = gammas[selection_mask(
                gammas, gammaness=gammaness_wave, r_max=r_max_gamm_wave)]
        proton = proton[selection_mask(
                proton, gammaness=gammaness_wave, r_max=r_max_prot)]

    print()
    print("gammas selected (wavelets):", len(gammas))
    print("proton selected (wavelets):", len(proton))

    SensCalc = SensitivityPointSource(
                reco_energies={'g': gammas['reco_Energy'].values*u.TeV,
                               'p': proton['reco_Energy'].values*u.TeV},
                mc_energies={'g': gammas['MC_Energy'].values*u.TeV,
                             'p': proton['MC_Energy'].values*u.TeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton},
                flux_unit=flux_unit)

    sensitivities = SensCalc.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': e_minus_2, 'p': e_minus_2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm_wave/r_max_prot)**2,
                            # sensitivity_energy_bin_edges=
                            #     10**np.array([-1, -.75, -.5, -.25, 0, 2,
                            #               2.25, 2.5, 2.75, 3, 9])*u.TeV
                                )
    weights = SensCalc.event_weights

    NExpGammas = sum(SensCalc.exp_events_per_energy_bin['g'])
    NExpProton = sum(SensCalc.exp_events_per_energy_bin['p'])

    print()
    print("expected gammas (wavelets):", NExpGammas)
    print("expected proton (wavelets):", NExpProton)

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "proton", "tail"))

    if False:
        fig = plt.figure()
        tax = plt.subplot(121)
        histo = np.histogram2d(gammas_t["NTels_reco"], gammas_t["gammaness"],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                        aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
        cb = fig.colorbar(im, ax=tax)
        tax.set_title("gammas")
        tax.set_xlabel("NTels")
        tax.set_ylabel("gammaness")

        tax = plt.subplot(122)
        histo = np.histogram2d(proton_t["NTels_reco"], proton_t["gammaness"],
                               bins=(range(1, 10), np.linspace(0, 1, 11)))[0].T
        histo_normed = histo / histo.max(axis=0)
        im = tax.imshow(histo_normed, interpolation='none', origin='lower',
                        aspect='auto', extent=(1, 9, 0, 1), cmap=plt.cm.inferno)
        cb = fig.colorbar(im, ax=tax)
        tax.set_title("protons")
        tax.set_xlabel("NTels")
        tax.set_ylabel("gammaness")

        plt.show()

    print()
    print("gammas present (tailcuts):", len(gammas_t))
    print("proton present (tailcuts):", len(proton_t))

    # applying some cuts
    if apply_cuts:
        gammas_t = gammas_t[selection_mask(
                gammas_t, gammaness=gammaness_tail, r_max=r_max_gamm_tail)]
        proton_t = proton_t[selection_mask(
                proton_t, gammaness=gammaness_tail, r_max=r_max_prot)]

    print()
    print("gammas selected (tailcuts):", len(gammas_t))
    print("proton selected (tailcuts):", len(proton_t))

    SensCalc_t = SensitivityPointSource(
                reco_energies={'g': gammas_t['reco_Energy'].values*u.TeV,
                               'p': proton_t['reco_Energy'].values*u.TeV},
                mc_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                             'p': proton_t['MC_Energy'].values*u.GeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton},
                flux_unit=flux_unit)

    sensitivities_t = SensCalc_t.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': e_minus_2, 'p': e_minus_2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm_tail/r_max_prot)**2)

    weights_t = SensCalc_t.event_weights

    gammas_t["weights"] = weights_t['g']
    proton_t["weights"] = weights_t['p']

    NExpGammas_t = sum(SensCalc_t.exp_events_per_energy_bin['g'])
    NExpProton_t = sum(SensCalc_t.exp_events_per_energy_bin['p'])

    print()
    print("expected gammas (tailcuts):", NExpGammas_t)
    print("expected proton (tailcuts):", NExpProton_t)

    # do some plotting
    if args.plot:
        make_sensitivity_plots(SensCalc, SensCalc_t,
                               sensitivities, sensitivities_t)


def main_xi68_cut(percentile={'w': 68, 't': 68}, res_scale=1):

    def selection_mask(event_table, ntels=3, gammaness=.75, r_max=None):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness) &
                (event_table["off_angle"] < r_max(event_table["reco_Energy"])))

    apply_cuts = True
    gammaness_wave = .95
    gammaness_tail = .95
    theta_on_off_ratio = 6

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * (498-14)
    NProton_simulated = NProton_per_File * (6998-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "gamma", "wave"))

    proton = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "proton", "wave"))

    print()
    print("gammas present (wavelets):", len(gammas))
    print("proton present (wavelets):", len(proton))

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}.h5".format(
                    args.events_dir, args.in_file, "proton", "tail"))
    print()
    print("gammas present (tailcuts):", len(gammas_t))
    print("proton present (tailcuts):", len(proton_t))

    # define edges to sort events in
    n_e_bins = 20
    e_bins_fine = np.logspace(-1, np.log10(600), n_e_bins)*u.TeV
    xi_ebinned_w = [[] for a in range(n_e_bins)]
    xi_ebinned_t = [[] for a in range(n_e_bins)]

    # sort off-angles in bins of recnstructed energy
    for xi, en in zip(gammas["off_angle"], gammas["reco_Energy"]):
        xi_ebinned_w[np.digitize(en, e_bins_fine)].append(xi)
    for xi, en in zip(gammas_t["off_angle"], gammas_t["reco_Energy"]):
        xi_ebinned_t[np.digitize(en, e_bins_fine)].append(xi)

    # get the 68th percentile resolution in every energy bin
    xi_cut_ebinned_w = np.full(len(xi_ebinned_w), np.inf)
    xi_cut_ebinned_t = np.full(len(xi_ebinned_t), np.inf)
    for i, (ebin_w, ebin_t) in enumerate(zip(xi_ebinned_w, xi_ebinned_t)):
        try:
            xi_cut_ebinned_w[i] = np.percentile(ebin_w, percentile['w'])
            xi_cut_ebinned_t[i] = np.percentile(ebin_t, percentile['t'])
        except IndexError:
            pass

    from scipy.optimize import curve_fit
    popt_w, pcov_w = curve_fit(fitfunc_log,
                               e_bins_fine[xi_cut_ebinned_w != np.inf].value,
                               xi_cut_ebinned_w[xi_cut_ebinned_w != np.inf])
    popt_t, pcov_t = curve_fit(fitfunc_log,
                               e_bins_fine[xi_cut_ebinned_t != np.inf].value,
                               xi_cut_ebinned_t[xi_cut_ebinned_t != np.inf])

    # print("fit args w:", popt_w)
    # print("fit args t:", popt_t)

    # fit args w: [ 0.34308449 -0.61659649  0.47705377  0.05055861 -0.18217412  0.04727536
    #   0.01758018 -0.00635498]
    # fit args t: [ 0.30052412 -0.26593548 -0.37978748 -0.29189531  2.39562965 -2.45684319
    #   0.95933452 -0.13139701]

    if True:
        plt.figure()
        plt.semilogx(e_bins_fine[1:-1], xi_cut_ebinned_w[1:-1],
                     color="darkred", marker="^", ls="",
                     label="MC wave -- {} %".format(percentile['w']))
        plt.semilogx(e_bins_fine[1:-1], xi_cut_ebinned_t[1:-1],
                     color="darkorange", marker="^", ls="",
                     label="MC tail -- {} %".format(percentile['t']))
        plt.semilogx(e_bins_fine[1:-1], fitfunc_log(e_bins_fine[1:-1].value, *popt_w),
                     color="darkred", label="fit wave")
        plt.semilogx(e_bins_fine[1:-1], fitfunc_log(e_bins_fine[1:-1].value, *popt_t),
                     color="darkorange", label="fit tail")
        plt.xlabel("E / TeV")
        plt.ylabel(r"$\xi_{cut}$ / deg")
        plt.gca().set_yscale("log")
        plt.grid()
        plt.legend()
        plt.pause(.1)

    # applying some cuts
    if apply_cuts:
        gammas = gammas[selection_mask(
                gammas, gammaness=gammaness_wave,
                r_max=lambda e: fitfunc_log(e, *popt_w)*res_scale)]
        proton = proton[selection_mask(
                proton, gammaness=gammaness_wave,
                r_max=lambda e: fitfunc_log(e, *popt_w)*res_scale*theta_on_off_ratio)]
        gammas_t = gammas_t[selection_mask(
                gammas_t, gammaness=gammaness_tail,
                r_max=lambda e: fitfunc_log(e, *popt_t)*res_scale)]
        proton_t = proton_t[selection_mask(
                proton_t, gammaness=gammaness_tail,
                r_max=lambda e: fitfunc_log(e, *popt_t)*res_scale*theta_on_off_ratio)]

    print()
    print("gammas selected (wavelets):", len(gammas))
    print("proton selected (wavelets):", len(proton))
    print()
    print("gammas selected (tailcuts):", len(gammas_t))
    print("proton selected (tailcuts):", len(proton_t))

    SensCalc = SensitivityPointSource(
            reco_energies={'g': gammas['reco_Energy'].values*u.TeV,
                           'p': proton['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas['MC_Energy'].values*u.GeV,
                         'p': proton['MC_Energy'].values*u.GeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    event_weights = SensCalc.generate_event_weights(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2})

    sensitivities = SensCalc.get_sensitivity(
            alpha=theta_on_off_ratio**-2,
            sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    # sensitvity for tail cuts
    SensCalc_t = SensitivityPointSource(
            reco_energies={'g': gammas_t['reco_Energy'].values*u.TeV,
                           'p': proton_t['reco_Energy'].values*u.TeV},
            mc_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                         'p': proton_t['MC_Energy'].values*u.GeV},
            energy_bin_edges={'g': edges_gammas,
                              'p': edges_proton},
            flux_unit=flux_unit)

    event_weights_t = SensCalc_t.generate_event_weights(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': cr_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2})

    sensitivities_t = SensCalc_t.get_sensitivity(
            alpha=theta_on_off_ratio**-2,
            sensitivity_energy_bin_edges=sensitivity_energy_bin_edges)

    make_performance_plots(gammas, proton, suptitle="wavelet")
    # make_performance_plots(gammas_t, proton_t, suptitle="tailcut")

    make_sensitivity_plots(SensCalc, SensCalc_t,
                           sensitivities, sensitivities_t)


def main_minimise():
    theta_on_off_ratio = 4.5

    def selection_mask(event_table, ntels=3, gammaness=None, r_max=None):
        return ((event_table["NTels_reco"] >= ntels) &
                (event_table["gammaness"] > gammaness(event_table["reco_Energy"])) &
                (event_table["off_angle"] < r_max(event_table["reco_Energy"])))

    def make_sensitivity(gammas, proton, sens_bins=None):

        SensCalc = SensitivityPointSource(
                reco_energies={'g': gammas['reco_Energy'].values*u.TeV,
                               'p': proton['reco_Energy'].values*u.TeV},
                mc_energies={'g': gammas['MC_Energy'].values*u.GeV,
                             'p': proton['MC_Energy'].values*u.GeV},
                energy_bin_edges={'g': edges_gammas,
                                  'p': edges_proton},
                flux_unit=flux_unit)

        event_weights = SensCalc.generate_event_weights(
                                n_simulated_events={'g': NGammas_simulated,
                                                    'p': NProton_simulated},
                                generator_areas={'g': np.pi * (1000*u.m)**2,
                                                 'p': np.pi * (2000*u.m)**2},
                                observation_time=observation_time,
                                spectra={'g': crab_source_rate,
                                         'p': cr_background_rate},
                                e_min_max={"g": (0.1, 330)*u.TeV,
                                           "p": (0.1, 600)*u.TeV},
                                generator_gamma={"g": 2, "p": 2})

        sens = SensCalc.get_sensitivity(
                alpha=theta_on_off_ratio**-2,
                sensitivity_energy_bin_edges=sens_bins)

        return sens, SensCalc

    class NoSignalException(Exception):
        pass

    def sensitivity_minimiser(X, gammas, proton, sens_bin):

        r_max, gammaness = X
        n_tels = 3

        fit_gammas = gammas[(sens_bin[0].value < gammas["reco_Energy"]) &
                            (gammas["reco_Energy"] < sens_bin[1].value)]
        fit_proton = proton[(sens_bin[0].value < proton["reco_Energy"]) &
                            (proton["reco_Energy"] < sens_bin[1].value)]

        if not len(fit_gammas):
            raise NoSignalException

        fit_gammas = fit_gammas[(fit_gammas["NTels_reco"] >= n_tels) &
                                (fit_gammas["gammaness"] > gammaness) &
                                (fit_gammas["off_angle"] < r_max)]
        fit_proton = fit_proton[(fit_proton["NTels_reco"] >= n_tels) &
                                (fit_proton["gammaness"] > gammaness) &
                                (fit_proton["off_angle"] < r_max*theta_on_off_ratio)]

        try:
            sens = make_sensitivity(fit_gammas, fit_proton,
                                    sens_bin)[0]["Sensitivity_base"][0]
        except:
            sens = np.inf

        print()
        print(X)
        print("N gammas:", len(fit_gammas))
        print("N proton:", len(fit_proton))
        print("sensitvity:", sens)

        return sens

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * (498-14)
    NProton_simulated = NProton_per_File * (6998-100)

    print()
    print("gammas simulated:", NGammas_simulated)
    print("proton simulated:", NProton_simulated)
    print()
    print("observation time:", observation_time)

    gammas = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1001-run1012.h5".format(
                    args.events_dir, args.in_file, "gamma", "wave"))

    proton = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10000-run10043.h5".format(
                    args.events_dir, args.in_file, "proton", "wave"))

    print()
    print("gammas present (wavelets):", len(gammas))
    print("proton present (wavelets):", len(proton))

    # now for tailcut
    gammas_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run1015-run1026.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10100-run10143.h5".format(
                    args.events_dir, args.in_file, "proton", "tail"))
    print()
    print("gammas present (tailcuts):", len(gammas_t))
    print("proton present (tailcuts):", len(proton_t))

    # faking reconstructed energy
    gammas["reco_Energy"] = gammas["MC_Energy"] * u.GeV.to(u.TeV)
    proton["reco_Energy"] = np.random.uniform(.100, 600, len(proton))*u.TeV
    gammas_t["reco_Energy"] = gammas_t["MC_Energy"] * u.GeV.to(u.TeV)
    proton_t["reco_Energy"] = np.random.uniform(.100, 600, len(proton_t))*u.TeV

    # finding the optimal theta and gammaness cuts
    en_cuts, xi_cuts, ga_cuts = [], [], []
    sens_bins = np.logspace(-1, 3, 17)*u.TeV
    for i in range(sens_bins.shape[0]-1):
        # if i not in [4, 3]: continue
        sens_bin = sens_bins[i:i+2]
        min_sens = np.inf
        try:
            for xi in np.logspace(-2, .5, 20):
                for gam in np.linspace(.1, 1, 20):
                    sens = sensitivity_minimiser((xi, gam), gammas, proton,
                                                 sens_bin)
                    if sens < min_sens:
                        min_sens = sens
                        min_xi = xi
                        min_gam = gam
            print("minimum:", min_xi, min_gam, min_sens)
            en_cuts.append(np.diff(sens_bin)[0])
            xi_cuts.append(min_xi)
            ga_cuts.append(min_gam)

        except NoSignalException:
            print("no signal in this energy bin")
            continue

    # from scipy.optimize import minimize
    # res = minimize(sensitivity_minimiser, [min_xi, min_gam],
    #                args=(gammas, proton, sens_bin),
    #                method='SLSQP', bounds=[(0, 3), (0, .95)],
    #             #    method='L-BFGS-B', bounds=[(0, 3), (0, 1)],
    #                options={'disp': False}
    #                ).x
    # print(res)

    en_cuts = np.array([en/u.TeV for en in en_cuts])*u.TeV
    print(en_cuts)
    print(xi_cuts)
    print(ga_cuts)

    # fitting polynomials to the optimal theta and gammaness cuts
    from scipy.optimize import curve_fit
    popt_x, pcov_x = curve_fit(fitfunc_log, en_cuts.value, xi_cuts)
    popt_g, pcov_g = curve_fit(fitfunc_log, en_cuts.value, ga_cuts)

    popt_x = np.array([0.34308449, -0.61659649, 0.47705377, 0.05055861, -0.18217412,
                       0.04727536, 0.01758018, -0.00635498])

    plt.figure()
    plt.semilogx(en_cuts, xi_cuts, label="xi cuts")
    plt.semilogx(en_cuts, fitfunc_log(en_cuts.value, *popt_x),
                 marker='^', ls='', label="xi fit")
    plt.semilogx(en_cuts, ga_cuts, label="ga cuts")
    plt.semilogx(en_cuts, fitfunc_log(en_cuts.value, *popt_g),
                 marker='^', ls='', label="ga fit")
    plt.legend()
    plt.pause(.1)

    # finally applying the cuts
    gammas = gammas[selection_mask(gammas,
                                   gammaness=lambda e: fitfunc_log(e, *popt_g),
                                   r_max=lambda e: fitfunc_log(e, *popt_x))]
    proton = proton[selection_mask(proton,
                                   gammaness=lambda e: fitfunc_log(e, *popt_g),
                                   r_max=lambda e: fitfunc_log(e, *popt_x)
                                   * theta_on_off_ratio)]

    sens, SensCalc = make_sensitivity(gammas, proton,
                                      np.logspace(-1, 3, 17)*u.TeV)
    make_sensitivity_plots(SensCalc, SensCalc, sens, sens)


def make_sensitivity_plots(SensCalc, SensCalc_t, sensitivities, sensitivities_t):
        bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.
        bin_centres_p = (edges_proton[1:]+edges_proton[:-1])/2.

        bin_widths_g = np.diff(edges_gammas.value)
        bin_widths_p = np.diff(edges_proton.value)

        if args.verbose:
            # plot MC generator spectrum and selected spectrum
            plt.figure()
            plt.subplot(121)
            plt.bar(bin_centres_g.value,
                    SensCalc_t.generator_energy_hists['g'], label="generated",
                    align="center", width=bin_widths_g)
            plt.bar(bin_centres_g.value,
                    SensCalc_t.selected_events['g'], label="selected",
                    align="center", width=bin_widths_g)
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("number of events")
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.title("gammas -- tailcuts")
            plt.legend()

            plt.subplot(122)
            plt.bar(bin_centres_p.value,
                    SensCalc_t.generator_energy_hists['p'], label="generated",
                    align="center", width=bin_widths_p)
            plt.bar(bin_centres_p.value,
                    SensCalc_t.selected_events['p'], label="selected",
                    align="center", width=bin_widths_p)
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("number of events")
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")
            plt.title("protons -- tailcuts")
            plt.legend()

            # plot the number of expected events in each energy bin
            plt.figure()
            plt.bar(
                    bin_centres_p.value,
                    SensCalc_t.exp_events_per_energy_bin['p'], label="proton",
                    align="center", width=np.diff(edges_proton.value), alpha=.75)
            plt.bar(
                    bin_centres_g.value,
                    SensCalc_t.exp_events_per_energy_bin['g'], label="gamma",
                    align="center", width=np.diff(edges_gammas.value), alpha=.75)
            plt.gca().set_xscale("log")
            plt.gca().set_yscale("log")

            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.legend()

            # plot effective area
            plt.figure(figsize=(16, 8))
            plt.suptitle("ASTRI Effective Areas")
            plt.subplot(121)
            plt.loglog(
                bin_centres_g,
                SensCalc.effective_areas['g'], label="wavelets")
            plt.loglog(
                bin_centres_g,
                SensCalc_t.effective_areas['g'], label="tailcuts")
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_g.unit)+"}$")
            plt.ylabel(r"effective area / $\mathrm{m^2}$")
            plt.title("gammas")
            plt.legend()

            plt.subplot(122)
            plt.loglog(
                bin_centres_p,
                SensCalc.effective_areas['p'], label="wavelets")
            plt.loglog(
                bin_centres_p,
                SensCalc_t.effective_areas['p'], label="tailcuts")
            plt.xlabel(r"$E_\mathrm{MC} / \mathrm{"+str(bin_centres_p.unit)+"}$")
            plt.ylabel(r"effective area / $\mathrm{m^2}$")
            plt.title("protons")
            plt.legend()

            # plot the angular distance of the reconstructed shower direction
            # from the pseudo-source

            figure = plt.figure()
            bins = 60

            plt.subplot(211)
            plt.hist([proton_t['off_angle']**2,
                      gammas_t["off_angle"]**2],
                     weights=[weights_t['p'], weights_t['g']],
                     rwidth=1, stacked=True,
                     range=(0, .3), label=("protons", "gammas"),
                     log=False, bins=bins)
            plt.xlabel(r"$(\vartheta/^\circ)^2$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.xlim([0, .3])
            plt.legend(loc="upper right", title="tailcuts")

            plt.subplot(212)
            plt.hist([proton['off_angle']**2,
                      gammas["off_angle"]**2],
                     weights=[weights['p'], weights['g']],
                     rwidth=1, stacked=True,
                     range=(0, .3), label=("protons", "gammas"),
                     log=False, bins=bins)
            plt.xlabel(r"$(\vartheta/^\circ)^2$")
            plt.ylabel("expected events in {}".format(observation_time))
            plt.xlim([0, .3])
            plt.legend(loc="upper right", title="wavelets")
            plt.tight_layout()

            if args.write:
                save_fig("plots/theta_square")

        # the point-source sensitivity binned in energy

        plt.figure()
        # draw the crab flux as well
        crab_bins = np.logspace(-1, 3, 17)
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

        # plt.semilogy(
        #     sensitivities["Energy"],
        #     (sensitivities["Sensitivity"].to(flux_unit) *
        #      sensitivities["Energy"].to(u.erg)**2),
        #     color="darkred",
        #     marker="s",
        #     label="wavelets")
        plt.semilogy(
            sensitivities["Energy"].to(u.TeV),
            (sensitivities["Sensitivity_base"].to(flux_unit) *
             sensitivities["Energy"].to(u.erg)**2),
            color="darkgreen",
            marker="^",
            # ls="",
            label="wavelets (no upscale)")

        # plt.semilogy(
        #     sensitivities_t["Energy"].to(energy_unit),
        #     (sensitivities_t["Sensitivity"].to(flux_unit) *
        #      sensitivities_t["Energy"].to(u.erg)**2),
        #     color="C0",
        #     marker="s",
        #     label="tailcuts")
        plt.semilogy(
            sensitivities_t["Energy"].to(energy_unit),
            (sensitivities_t["Sensitivity_base"].to(flux_unit) *
             sensitivities_t["Energy"].to(u.erg)**2),
            color="darkorange",
            marker="^",
            # ls="",
            label="tailcuts (no upscale)")

        plt.legend(title="Obsetvation Time: {}".format(observation_time))
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
        plt.gca().set_xscale("log")
        plt.grid()
        plt.xlim([1e-2, 2e3])
        plt.ylim([5e-15, 5e-10])

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

        # this demonstrates how to flatten the proton distribution in the theta plot:
        #     NProtons = np.sum(proton['off_angle'][(proton['off_angle'].values**2) < 10])
        #     proton_weight_flat = np.ones(bins) * NProtons/bins
        #     proton_angle_flat = np.linspace(0, 10, bins, False)
        #     proton_angle = proton_angle_flat
        #     proton_weight = proton_weight_flat


def make_performance_plots(gammas, proton, suptitle=None):

    fig, axes = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.11, right=0.97, hspace=0.39, wspace=0.29)
    plot_hex_and_violin(gammas["NTels_reco"], np.log10(gammas["off_angle"]),
                        np.arange(1, 11)-.5,
                        xlabel="N Tels", ylabel=r"$\log_{10}(\xi$ / degree)",
                        do_hex=False, axis=axes[0], extent=[1.5, 8, -3, 0])
    plot_hex_and_violin(np.log10(gammas["reco_Energy"]),
                        np.log10(gammas["off_angle"]),
                        np.linspace(-1, 3, 17),
                        xlabel="log_10(E_reco / TeV)",
                        ylabel=r"$\log_{10}(\xi$ / degree)",
                        v_padding=0.015, axis=axes[1], extent=[-.5, 2.5, -3, 0])
    if suptitle:
        plt.suptitle(suptitle)

    # plot resolution as energy-binned 68th percentiles
    percentile = {"g": 68, "p": 68}

    # define edges to sort events in
    n_e_bins = 20
    e_bins_fine = np.logspace(-1, np.log10(600), n_e_bins)*u.TeV
    xi_ebinned_g = [[] for a in range(n_e_bins)]
    xi_ebinned_p = [[] for a in range(n_e_bins)]

    # for the angular resolution

    # sort off-angles in bins of reconstructed energy
    for xi, en in zip(gammas["off_angle"], gammas["reco_Energy"]):
        xi_ebinned_g[np.digitize(en, e_bins_fine)].append(xi)
    for xi, en in zip(proton["off_angle"], proton["reco_Energy"]):
        xi_ebinned_p[np.digitize(en, e_bins_fine)].append(xi)

    # get the 68th percentile resolution in every energy bin
    xi68_ebinned_g = np.full(len(xi_ebinned_g), np.inf)
    xi68_ebinned_p = np.full(len(xi_ebinned_p), np.inf)
    for i, (ebin_g, ebin_p) in enumerate(zip(xi_ebinned_g, xi_ebinned_p)):
        try:
            xi68_ebinned_g[i] = np.percentile(ebin_g, percentile['g'])
            xi68_ebinned_p[i] = np.percentile(ebin_p, percentile['p'])
        except IndexError:
            pass

    plt.figure()
    plt.semilogx(e_bins_fine[1:-1], xi68_ebinned_g[1:-1],
                 color="darkred", marker="^", ls="-",
                 label="gamma -- {} %".format(percentile['g']))
    plt.semilogx(e_bins_fine[1:-1], xi68_ebinned_p[1:-1],
                 color="darkorange", marker="^", ls="--",
                 label="proton -- {} %".format(percentile['p']))
    plt.xlabel(r"$E_{reco}$ / TeV")
    plt.ylabel(r"$\xi_{68}$ / deg")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.legend()
    if suptitle:
        plt.suptitle(suptitle)

    # for the energy resolution
    n_e_bins = 50
    e_bins_fine = np.logspace(-1, np.log10(600), n_e_bins)*u.TeV

    # MC Energy vs. reco Energy 2D histograms
    fig = plt.figure()
    ax = plt.subplot(121)
    counts_g, _, _ = np.histogram2d(gammas["reco_Energy"],
                                    gammas["MC_Energy"],
                                    bins=(e_bins_fine, e_bins_fine))
    ax.pcolormesh(e_bins_fine.value, e_bins_fine.value, counts_g)
    plt.plot(e_bins_fine.value[[0, -1]], e_bins_fine.value[[0, -1]],
             color="darkgreen")
    plt.title("gamma")
    plt.xlabel(r"$E_{MC}$ / TeV")
    plt.ylabel(r"$E_{reco}$ / TeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid()

    ax = plt.subplot(122)
    counts_p, _, _ = np.histogram2d(proton["reco_Energy"],
                                    proton["MC_Energy"],
                                    bins=(e_bins_fine, e_bins_fine))
    ax.pcolormesh(e_bins_fine.value, e_bins_fine.value, counts_p)
    plt.plot(e_bins_fine.value[[0, -1]], e_bins_fine.value[[0, -1]],
             color="darkgreen")
    plt.title("proton")
    plt.xlabel(r"$E_{MC}$ / TeV")
    plt.ylabel(r"$E_{reco}$ / TeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid()

    plt.subplots_adjust(top=.90, bottom=.11, left=.12, right=.90,
                        hspace=.20, wspace=.31)

    if suptitle:
        plt.suptitle(suptitle)

    # energy resolution as 68th percentile of the relative reconstructed error binned in
    # reconstructed energy
    plt.figure()
    DeltaE = np.abs(gammas["reco_Energy"] - gammas["MC_Energy"])
    DeltaE_ebinned = [[] for a in range(n_e_bins)]

    # sort relative energy error in bins of reconstructed energy
    for DE, en_r, en_mc in zip(DeltaE, gammas["reco_Energy"], gammas["MC_Energy"]):
        DeltaE_ebinned[np.digitize(en_r, e_bins_fine)].append(DE/en_mc)

    # get the 68th percentile resolution in every energy bin
    DeltaE68_ebinned = np.full(len(DeltaE_ebinned), np.inf)
    for i, ebin in enumerate(DeltaE_ebinned):
        try:
            DeltaE68_ebinned[i] = np.percentile(ebin, 68)
        except IndexError:
            pass

    plt.plot(e_bins_fine, DeltaE68_ebinned, marker='^', color="darkgreen")
    plt.title("gamma")
    plt.xlabel(r"$E_{reco}$ / TeV")
    plt.ylabel(r"$(E_{reco} - E_{MC})/E_{MC}$")
    plt.gca().set_xscale("log")

    if suptitle:
        plt.suptitle(suptitle)


if __name__ == "__main__":
    np.random.seed(19)

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/events")
    parser.add_argument('--in_file', type=str, default="classified_events")
    args = parser.parse_args()

    # main_minimise()
    # main_const_theta_cut()

    main_xi68_cut(percentile={'w': 50, 't': 50})

    if args.plot:
        plt.show()
