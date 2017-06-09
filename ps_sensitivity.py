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

from ctapipe.analysis.sensitivity import (SensitivityPointSource,
                                          crab_source_rate, CR_background_rate, Eminus2)

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
# plt.style.use('t_slides')


# MC energy ranges:
# gammas: 0.1 to 330 TeV
# proton: 0.1 to 600 TeV
edges_gammas = np.logspace(2, np.log10(330000), 28) * u.GeV
edges_proton = np.logspace(2, np.log10(600000), 30) * u.GeV

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


def selection_mask(event_table, ntels=3, gammaness=.75, r_max=0.1*u.deg):
    return ((event_table["NTels_reco"] >= ntels) &
            (event_table["gammaness"] > gammaness) &
            (event_table["off_angle"] < r_max))


if __name__ == "__main__":

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/events")
    parser.add_argument('--in_file', type=str, default="classified_events")
    args = parser.parse_args()

    apply_cuts = True
    gammaness_wave = .75
    gammaness_tail = .75
    r_max_prot = 2*u.deg
    r_max_gamm = 0.05*u.deg

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

    # applying some cuts
    if apply_cuts:
        gammas = gammas[selection_mask(
                gammas, gammaness=gammaness_wave, r_max=r_max_gamm)]
        proton = proton[selection_mask(
                proton, gammaness=gammaness_wave, r_max=r_max_prot)]

    print()
    print("gammas selected (wavelets):", len(gammas))
    print("proton selected (wavelets):", len(proton))

    SensCalc = SensitivityPointSource(
                    mc_energies={'g': gammas['MC_Energy'].values*u.GeV,
                                 'p': proton['MC_Energy'].values*u.GeV},
                    energy_bin_edges={'g': edges_gammas,
                                      'p': edges_proton},
                    flux_unit=flux_unit)

    sensitivities = SensCalc.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': Eminus2, 'p': Eminus2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': CR_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm/r_max_prot)**2,
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
            "{}/{}_{}_{}_run1015-run1026.h5".format(
                    args.events_dir, args.in_file, "gamma", "tail"))

    proton_t = open_pytable_as_pandas(
            "{}/{}_{}_{}_run10100-run10143.h5".format(
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
                gammas_t, gammaness=gammaness_tail, r_max=r_max_gamm)]
        proton_t = proton_t[selection_mask(
                proton_t, gammaness=gammaness_tail, r_max=r_max_prot)]

    print()
    print("gammas selected (tailcuts):", len(gammas_t))
    print("proton selected (tailcuts):", len(proton_t))

    SensCalc_t = SensitivityPointSource(
                    mc_energies={'g': gammas_t['MC_Energy'].values*u.GeV,
                                 'p': proton_t['MC_Energy'].values*u.GeV},
                    energy_bin_edges={'g': edges_gammas,
                                      'p': edges_proton},
                    flux_unit=flux_unit)

    sensitivities_t = SensCalc_t.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': Eminus2, 'p': Eminus2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                             'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            spectra={'g': crab_source_rate,
                                     'p': CR_background_rate},
                            e_min_max={"g": (0.1, 330)*u.TeV,
                                       "p": (0.1, 600)*u.TeV},
                            generator_gamma={"g": 2, "p": 2},
                            alpha=(r_max_gamm/r_max_prot)**2,
                            # sensitivity_energy_bin_edges=
                            #     10**np.array([-1, -.75, -.5, -.25, 0, 2,
                            #                   2.25, 2.5, 2.75, 3, 9])*u.TeV
                                )
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
        #     sensitivities["MC Energy"],
        #     (sensitivities["Sensitivity"].to(flux_unit) *
        #      sensitivities["MC Energy"].to(u.erg)**2),
        #     color="darkred",
        #     marker="s",
        #     label="wavelets")
        plt.semilogy(
            sensitivities["MC Energy"].to(u.TeV),
            (sensitivities["Sensitivity_base"].to(flux_unit) *
             sensitivities["MC Energy"].to(u.erg)**2),
            color="darkgreen",
            marker="^",
            # ls="",
            label="wavelets (no upscale)")

        # plt.semilogy(
        #     sensitivities_t["MC Energy"].to(energy_unit),
        #     (sensitivities_t["Sensitivity"].to(flux_unit) *
        #      sensitivities_t["MC Energy"].to(u.erg)**2),
        #     color="C0",
        #     marker="s",
        #     label="tailcuts")
        plt.semilogy(
            sensitivities_t["MC Energy"].to(energy_unit),
            (sensitivities_t["Sensitivity_base"].to(flux_unit) *
             sensitivities_t["MC Energy"].to(u.erg)**2),
            color="darkorange",
            marker="^",
            # ls="",
            label="tailcuts (no upscale)")

        plt.legend(title="Obsetvation Time: {}".format(observation_time))
        plt.xlabel('E / {:latex}'.format(energy_unit))
        plt.ylabel(r'$E^2 \Phi /$ {:latex}'.format(sensitivity_unit))
        plt.gca().set_xscale("log")
        plt.xlim([1e-2, 2e3])

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

        plt.show()
