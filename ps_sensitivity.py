#!/usr/bin/env python3

import numpy as np

from astropy.table import Table
from astropy import units as u

from itertools import chain

from helper_functions import *

from ctapipe.analysis.Sensitivity import *
from ctapipe.analysis.Sensitivity import crab_source_rate, CR_background_rate, Eminus2

import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
plt.style.use('t_slides')

'''
MC energy ranges:
gammas: 0.1 to 330 TeV
proton: 0.1 to 600 TeV
'''
#edges_energy = np.linspace(0, 6, 28)
edges_gammas = np.linspace(2, np.log10(330000), 14)
edges_proton = np.linspace(2, np.log10(600000), 15)

angle_unit = u.deg
energy_unit = u.GeV
flux_unit = u.erg/(u.m**2*u.s)

observation_time = 50*u.h

# PyTables
import tables as tb

if __name__ == "__main__":

    parser = make_argparser()
    parser.add_argument('--events_dir', type=str, default="data/reconstructed_events")
    parser.add_argument('--in_file', type=str, default="classified_events")
    args = parser.parse_args()

    gammas_infile = tb.open_file(
            "{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "gamma", "wave"),
            mode='r')
    gammas_pyt = gammas_infile.root.reco_events

    proton_infile = tb.open_file(
            "{}/{}_{}_{}.h5".format(args.events_dir, args.in_file, "proton", "wave"),
            mode='r')
    proton_pyt = proton_infile.root.reco_event

    gammas = pd.DataFrame(gammas_pyt[:])
    proton = pd.DataFrame(proton_pyt[:])

    # applying some cuts
    if True:
        mask_g = (gammas["NTels_reco"] > 3) & (gammas["gamma_ratio"] > .75)
        mask_p = (proton["NTels_reco"] > 3) & (proton["gamma_ratio"] > .75)
        gammas = gammas[mask_g]
        proton = proton[mask_p]

    NGammas_selected = len(gammas['off_angle'])
    NProton_selected = len(proton['off_angle'])

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File * 20
    NProton_simulated = NProton_per_File * 101

    print("gammas selected / simulated", NGammas_selected, NGammas_simulated)
    print("proton selected / simulated", NProton_selected, NProton_simulated)

    SensCalc = Sensitivity_PointSource(
                        mc_energies={'g': gammas['EnMC'],
                                     'p': proton['EnMC']},
                        off_angles={'g': gammas['off_angle'].values*angle_unit,
                                    'p': proton['off_angle'].values*angle_unit},
                        energy_bin_edges={'g': edges_gammas,
                                        'p': edges_proton},
                        energy_unit=energy_unit, flux_unit=flux_unit)

    Eff_Areas = SensCalc.get_effective_areas(
                    n_simulated_events={'g': NGammas_simulated,
                                        'p': NProton_simulated},
                    generator_spectra={'g': Eminus2, 'p': Eminus2},
                    generator_areas={'g': np.pi * (1000*u.m)**2,
                                     'p': np.pi * (2000*u.m)**2})

    exp_events_per_E = SensCalc.get_expected_events(
        rates={'g': crab_source_rate, 'p': CR_background_rate},
        observation_time=observation_time)

    NExpGammas = sum(exp_events_per_E['g'])
    NExpProton = sum(exp_events_per_E['p'])

    print("expected gammas:", NExpGammas)
    print("expected proton:", NExpProton)

    if args.plot and args.verbose:
        plt.figure()
        plt.semilogy((edges_gammas[1:] + edges_gammas[:-1])/2,
                     Eff_Areas['g'], "b", label='Gammas')
        plt.semilogy((edges_proton[1:] + edges_proton[:-1])/2,
                     Eff_Areas['p'], "r", label='Protons')
        plt.title("Effective Area")
        plt.xlabel(r"$\log_{10}(E/\mathrm{GeV})$")
        plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
        plt.pause(.1)

    weights = SensCalc.scale_events_to_expected_events()
    sensitivities = SensCalc.get_sensitivity()

    # now for tailcut
    gammas_t = Table.read("data/selected_events/"
                          "selected_events_tail_g.fits")
    proton_t = Table.read("data/selected_events/"
                          "selected_events_tail_p.fits")

    SensCalc_t = Sensitivity_PointSource(
                    mc_energies={'g': gammas_t['MC_Energy'],
                                 'p': proton_t['MC_Energy']},
                    off_angles={'g': gammas_t['off_angles']*angle_unit,
                                'p': proton_t['off_angles']*angle_unit},
                    energy_bin_edges={'g': edges_gammas,
                                      'p': edges_proton},
                    energy_unit=energy_unit, flux_unit=flux_unit)

    sensitivities_t = SensCalc_t.calculate_sensitivities(
                            n_simulated_events={'g': NGammas_simulated,
                                                'p': NProton_simulated},
                            generator_spectra={'g': Eminus2, 'p': Eminus2},
                            generator_areas={'g': np.pi * (1000*u.m)**2,
                                            'p': np.pi * (2000*u.m)**2},
                            observation_time=observation_time,
                            rates={'g': Eminus2,
                                   'p': CR_background_rate})
    weights_t = SensCalc_t.event_weights

    # do some plotting
    if args.plot:
        bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.

        # plot effective area
        plt.figure()
        plt.loglog(
            10 ** edges_gammas[:-1],
            SensCalc.effective_areas['g'])
        plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
        plt.ylabel("effective area")
        plt.suptitle("gammas")
        plt.pause(.1)

        # the point-source sensitivity binned in energy
        plt.figure()
        plt.semilogy(
            sensitivities["Energy MC"],
            sensitivities["Sensitivity"],
            label=args.mode)
        plt.semilogy(
            sensitivities_t["Energy MC"],
            sensitivities_t["Sensitivity"],
            label="tail")
        plt.legend()
        plt.xlabel('E / {:latex}'.format(SensCalc.energy_unit))
        plt.gca().set_xscale("log")
        plt.ylabel(r'$\Phi E^2 /$ {:latex}'.format(SensCalc.flux_unit))
        plt.pause(.1)

        # plot a sky image of the events
        # useless since too few actual background events
        if False:
            fig2 = plt.figure()
            plt.hexbin(
                [(ph-180)*np.sin(th*u.deg) for
                    ph, th in zip(chain(gammas['phi'], proton_pyt['phi']),
                                  chain(gammas['theta'], proton_pyt['theta']))],
                [a for a in chain(gammas['theta'], proton_pyt['theta'])],
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
                save_fig("plots/skymap_{}".format(args.mode))

        '''
        plot the angular distance of the reconstructed shower direction
        from the pseudo source in different scales '''
        figure = plt.figure()
        #plt.subplot(211)
        #plt.hist([off_angles['p'], off_angles['g']],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(0, 5),
                 #bins=25)
        #plt.xlabel(r"$\vartheta / \mathrm{"+str(angle_unit)+"}$")
        #plt.ylabel("expected events in {}".format(ObsTime))
        ##plt.ylim([0, 3000])

        #if args.write:
            #tikz_save("plots/"+args.mode+"_proto_significance.tex",
                        #draw_rectangles=True)

        #plt.subplot(212)
        if True:
            bins = 50
            NProtons = np.sum(proton['off_angle'][(proton['off_angle'].values**2) < 10])
            proton_weight_flat = np.ones(bins) * NProtons/bins
            proton_angle_flat = np.linspace(0, 10, bins, False)
            proton_angle = proton_angle_flat
            proton_weight = proton_weight_flat
        else:
            proton_angle = proton['off_angle']**2
            proton_weight = weights['p']

        plt.hist([proton_angle,
                  gammas['off_angle'].values**2],
                 weights=[proton_weight, weights['g']], rwidth=1, stacked=True,
                 range=(0, 10), label=("protons", "gammas"),
                 log=True, bins=bins)
        plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        plt.ylim([1e-1, 1e6])
        plt.legend()
        plt.suptitle(args.mode)

        plt.pause(.1)
        #plt.subplot(313)
        #plt.hist([1-np.clip(np.cos(proton['off_angles']*u.degree.to(u.rad)), -1, 1-1e-6),
                  #1-np.clip(np.cos(gammas['off_angles']*u.degree.to(u.rad)), -1, 1-1e-6)],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(0, 5e-3),
                 #bins=30)
        #plt.xlabel(r"$1-\cos(\vartheta)$")
        #plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.xlim([0, 5e-3])

        #plt.tight_layout()

        #figure = plt.figure()
        #if True:
            #NProtons_t = np.sum(proton_t['off_angles'][(proton_t['off_angles']**2) < 10])
            #proton_weight_flat = np.ones(50) * NProtons_t/50
            #proton_angle_flat = np.linspace(0,10,50,False)
            #proton_angle_t = proton_angle_flat
            #proton_weight_t = proton_weight_flat
        #else:
            #proton_angle_t = proton_t['off_angles']**2
            #proton_weight_t = weights_t['p']

        #plt.hist([proton_angle_t,
                  #gammas_t['off_angles']**2],
                 #weights=[proton_weight_t, weights_t['g']], rwidth=1, stacked=True,
                 #range=(0, 10), label=("protons", "gammas"),
                 #log=True, bins=50)
        #plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        #plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.ylim([1e-1, 1e5])
        #plt.legend()
        #plt.suptitle("tail cuts (10 PE / 5 PE)")


        plt.show()





