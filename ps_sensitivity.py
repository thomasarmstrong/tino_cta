import numpy as np

from astropy.table import Table
from astropy import units as u

from itertools import chain

from helper_functions import *

from modules.Sensitivity import *

import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
plt.style.use('t_slides')

'''
MC energy ranges:
gammas: 0.1 to 330 TeV
proton: 0.1 to 600 TeV
'''
edges_gammas = np.concatenate((
    np.linspace(2, 2.5, 15, False),
    np.linspace(2.5, 3, 5, False),
    np.linspace(3, 3.5, 5, False),
    np.linspace(3.5, 4, 5, False),
    np.linspace(4, np.log10(330000), 5)
    ))
edges_proton = np.concatenate((
    np.linspace(2, 2.5, 1, False),
    np.linspace(2.5, 3, 4, False),
    np.linspace(3, 4, 3, False),
    np.linspace(4, 4.5, 5, False),
    np.linspace(4.5, np.log10(600000), 5)
    ))
NBins = 100

flux_unit = u.erg/(u.m**2*u.s)

# edges_gammas = np.linspace(np.log10(0.1*u.TeV.to(u.GeV)), np.log10(330*u.TeV.to(u.GeV)), NBins+1)
# edges_proton = np.linspace(np.log10(0.1*u.TeV.to(u.GeV)), np.log10(600*u.TeV.to(u.GeV)), NBins+1)

if __name__ == "__main__":

    args = make_argparser().parse_args()

    gammas = Table.read("data/selected_events/"
                        "selected_events_"+args.mode+"_g.fits")
    proton = Table.read("data/selected_events/"
                        "selected_events_"+args.mode+"_p.fits")

    NGammas_selected = len(gammas['off_angles'])
    NProton_selected = len(proton['off_angles'])

    NReuse_Gammas = 10
    NReuse_Proton = 20

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File *  9
    NProton_simulated = NProton_per_File * 51

    print("gammas selected / simulated", NGammas_selected, NGammas_simulated)
    print("proton selected / simulated", NProton_selected, NProton_simulated)

    SensCalc = Sensitivity_PointSource(gammas['MC_Energy'], proton['MC_Energy'],
                                       edges_gammas, edges_proton,
                                       flux_unit=flux_unit)

    Eff_Area_Gammas, Eff_Area_Proton = SensCalc.get_effective_areas(NGammas_simulated,
                                                                    NProton_simulated)

    exp_events_per_E_g, exp_events_per_E_p = \
        SensCalc.get_expected_events(source_rate=crab_source_rate)

    NExpGammas = sum(exp_events_per_E_g)
    NExpProton = sum(exp_events_per_E_p)

    print("expected gammas:", NExpGammas)
    print("expected proton:", NExpProton)

    if args.plot and args.verbose:
        plt.figure()
        plt.semilogy((edges_gammas[1:] + edges_gammas[:-1])/2,
                     Eff_Area_Gammas, "b", label='Gammas')
        plt.semilogy((edges_proton[1:] + edges_proton[:-1])/2,
                     Eff_Area_Proton, "r", label='Protons')
        plt.title("Effective Area")
        plt.xlabel(r"$\log_{10}(E/\mathrm{GeV})$")
        plt.ylabel(r"$A_\mathrm{eff} / \mathrm{m}^2$")
        plt.pause(.1)

    weight_g, weight_p = SensCalc.scale_events_to_expected_events()

    sensitivities = SensCalc.get_sensitivity(gammas['off_angles'], proton['off_angles'])

    # now for tailcut
    gammas_t = Table.read("data/selected_events/"
                          "selected_events_tail_g.fits")
    proton_t = Table.read("data/selected_events/"
                          "selected_events_tail_p.fits")

    SensCalc_t = Sensitivity_PointSource(gammas_t['MC_Energy'], proton_t['MC_Energy'],
                                         edges_gammas, edges_proton,
                                         flux_unit=flux_unit)
    SensCalc_t.get_effective_areas(NGammas_simulated, NProton_simulated)
    SensCalc_t.get_expected_events(source_rate=crab_source_rate)
    weight_g_t, weight_p_t = SensCalc_t.scale_events_to_expected_events()
    sensitivities_t = SensCalc_t.get_sensitivity(gammas_t['off_angles'],
                                                 proton_t['off_angles'])

    # do some plotting
    if args.plot:
        angle_unit = u.deg
        bin_centres_g = (edges_gammas[1:]+edges_gammas[:-1])/2.

        # the point-source sensitivity binned in energy
        plt.figure()
        plt.semilogy(
            [a[0] for a in sensitivities],
            [a[1] for a in sensitivities],
            label=args.mode)
        plt.semilogy(
            [a[0] for a in sensitivities_t],
            [a[1] for a in sensitivities_t],
            label="tail")
        plt.legend()
        plt.xlabel('log(E / GeV)')
        plt.ylabel(r'$\Phi E^2 /$ {:latex}'.format(SensCalc.flux_unit))
        plt.pause(.1)

        '''
        plot the source and spectra to be assumed in the sensitivity study '''
        if 0:
            flux_unit = u.erg / (u.cm**2 * u.s)
            figure = plt.figure()
            plt.subplot(121)
            plt.plot(
                     #  bin centres
                     (edges_gammas[1:]+edges_gammas[:-1])/2.,
                     (SourceRate  # event rate
                      #  times E²
                      * ((edges_gammas[1:]+edges_gammas[:-1])*u.GeV/2.)**2
                      #  divided by bin-width to make differential rate
                      / ((edges_gammas[1:]-edges_gammas[:-1])*u.GeV)).to(flux_unit),
                     marker='o')
            plt.yscale('log')
            plt.xlabel('log(E / GeV)')
            plt.ylabel(r"$E^2 \times \Phi / $[{0:latex}]".format(
                ((SourceRate*u.GeV).to(flux_unit)).unit
                ))
            plt.subplot(122)
            plt.plot((edges_proton[1:]+edges_proton[:-1])/2.,
                     (BackgrRate/((edges_proton[1:]-edges_proton[:-1])*u.GeV)),
                     marker='o')
            plt.yscale('log')
            #plt.show()
            plt.pause(.1)

        '''
        plot a sky image of the events
        MISSLEADING since not weighted to anything'''
        if False:
            fig2 = plt.figure()
            plt.hexbin(
                [(ph-180)*np.sin(th*u.deg) for
                    ph, th in zip(chain(gammas['phi'], proton['phi']),
                                  chain(gammas['theta'], proton['theta']))],
                [a for a in chain(gammas['theta'], proton['theta'])],
                gridsize=41, extent=[-2, 2, 18, 22],
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
            NProtons = np.sum(proton['off_angles'][(proton['off_angles']**2) < 10])
            proton_weight_flat = np.ones(bins) * NProtons/bins
            proton_angle_flat = np.linspace(0, 10, bins, False)
            proton_angle = proton_angle_flat
            proton_weight = proton_weight_flat
        else:
            proton_angle = proton['off_angles']**2
            proton_weight = weight_p

        plt.hist([proton_angle,
                  gammas['off_angles']**2],
                 weights=[proton_weight, weight_g], rwidth=1, stacked=True,
                 range=(0, 10), label=("protons", "gammas"),
                 bins=bins)
        plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        #plt.ylim([0, 5])
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

        figure = plt.figure()
        if True:
            NProtons_t = np.sum(proton_t['off_angles'][(proton_t['off_angles']**2) < 10])
            proton_weight_flat = np.ones(50) * NProtons_t/50
            proton_angle_flat = np.linspace(0,10,50,False)
            proton_angle_t = proton_angle_flat
            proton_weight_t = proton_weight_flat
        else:
            proton_angle_t = proton_t['off_angles']**2
            proton_weight_t = weight_p_t

        plt.hist([proton_angle_t,
                  gammas_t['off_angles']**2],
                 weights=[proton_weight_t, weight_g_t], rwidth=1, stacked=True,
                 range=(0, 10), label=("protons", "gammas"),
                 bins=50)
        plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        plt.ylabel("expected events in {}".format(SensCalc.observation_time))
        plt.ylim([0,5])
        plt.legend()
        plt.suptitle("tail cuts (10 PE / 5 PE)")


        plt.show()





