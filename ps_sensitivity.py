import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u

import random

from itertools import chain

from helper_functions import *


def crab_source_rate(E):
    '''
    Crab source rate:   dN/dE = 3e-7  * (E/TeV)**-2.48 / (TeV * m² * s)
    norm and spectral index reverse engineered from HESS plot... '''
    return 3e-7 * (E/u.TeV)**-2.48 / (u.TeV * u.m**2 * u.s)


def CR_background_rate(E):
    '''
    Cosmic Ray background rate: dN/dE = 0.215 * (E/TeV)**-8./3 / (TeV * m² * s * sr)
    norm and spectral index reverse engineered from "random" CR plot... '''
    return 100 * 0.1**(8./3) * (E/u.TeV)**(-8./3) / (u.TeV * u.m**2 * u.s * u.sr)


def Eminus2(e, unit=u.GeV):
    '''
    boring old E^-2 spectrum '''
    return (e/unit)**(-2) / (unit * u.s * u.m**2)

'''
MC energy ranges:
gammas: 0.1 to 330 TeV
proton: 0.1 to 600 TeV
'''
NBins=100
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
#edges_gammas = np.linspace(np.log10(0.1*u.TeV.to(u.GeV)), np.log10(330*u.TeV.to(u.GeV)), NBins+1)
#edges_proton = np.linspace(np.log10(0.1*u.TeV.to(u.GeV)), np.log10(600*u.TeV.to(u.GeV)), NBins+1)

if __name__ == "__main__":

    args = make_argparser().parse_args()

    gammas = Table.read("data/selected_events/"
                        "selected_events_"+args.mode+"_g.fits")
    proton = Table.read("data/selected_events/"
                        "selected_events_"+args.mode+"_p.fits")
    off_angles = {'p': proton['off_angles'],
                  'g': gammas['off_angles']}

    NGammas_selected = len(gammas['off_angles'])
    NProton_selected = len(proton['off_angles'])

    NReuse_Gammas = 10
    NReuse_Proton = 20

    ''' solid angle under which protons have been generated '''
    Omega_Proton = 2*np.pi*(1 - np.cos(6*u.deg))*u.rad**2

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File *  9
    NProton_simulated = NProton_per_File * 51

    Gen_Gammas = make_mock_event_rate(
                    [Eminus2], norm=[NGammas_simulated],
                    binEdges=edges_gammas)[0]
                    # Emin=0.1*u.TeV, Emax=330*u.TeV, NBins=NBins)

    Gen_Proton = make_mock_event_rate(
                    [Eminus2], norm=[NProton_simulated],
                    binEdges=edges_proton)[0]
                    # Emin=0.1*u.TeV, Emax=600*u.TeV, NBins=NBins)


    Sel_Gammas = np.histogram(np.log10(gammas['MC_Energy']), bins=edges_gammas)[0]
    Sel_Proton = np.histogram(np.log10(proton['MC_Energy']), bins=edges_proton)[0]

    Efficiency_Gammas = Sel_Gammas / Gen_Gammas
    Efficiency_Proton = Sel_Proton / Gen_Proton

    tot_Area_Gammas = np.pi * (1000*u.m)**2
    tot_Area_Proton = np.pi * (2000*u.m)**2

    Eff_Area_Gammas = Efficiency_Gammas * tot_Area_Gammas
    Eff_Area_Proton = Efficiency_Proton * tot_Area_Proton


    #fig = plt.figure()
    #plt.plot(edges_gammas[:-1], Efficiency_Gammas)
    #plt.title("Efficiency_Gammas")
    #plt.pause(.1)


    print("gammas selected / simulated", NGammas_selected, NGammas_simulated)
    print("proton selected / simulated", NProton_selected, NProton_simulated)

    SourceRate = make_mock_event_rate([crab_source_rate], binEdges=edges_gammas)[0]
    BackgrRate = make_mock_event_rate([CR_background_rate], binEdges=edges_proton)[0]
    BackgrRate *= Omega_Proton

    #fig = plt.figure()
    #plt.plot(edges_gammas[:-1], SourceRate)
    #plt.title("SourceRate")
    #plt.suptitle("sum: {}".format(sum(SourceRate)))
    #plt.pause(.1)

    SourceIntensity = 1
    ObsTime = (1*u.h)
    weight_vsE_g = SourceRate * ObsTime.to(u.s) * Eff_Area_Gammas * SourceIntensity
    weight_vsE_p = BackgrRate * ObsTime.to(u.s) * Eff_Area_Proton

    NExpGammas = sum(weight_vsE_g)
    NExpProton = sum(weight_vsE_p)

    #fig = plt.figure()
    #plt.plot(edges_gammas[:-1], weight_vsE_g)
    #plt.title("weight_vsE_g")
    #plt.suptitle("sum: {}".format(sum(weight_vsE_g)))
    #plt.pause(.1)


    weight_g = []
    weight_p = []
    for ev in gammas['MC_Energy']:
        weight_g.append(weight_vsE_g[np.digitize(np.log10(ev), edges_gammas) - 1])
    for ev in proton['MC_Energy']:
        weight_p.append(weight_vsE_p[np.digitize(np.log10(ev), edges_proton) - 1])

    weight_g = np.array(weight_g) / sum(weight_g) * NExpGammas
    weight_p = np.array(weight_p) / sum(weight_p) * NExpProton

    print("expected gammas:", NExpGammas)
    print("expected proton:", NExpProton)
    #off_angles['p'] = []
    #flat_weight = 100/NProtExp
    #weight_p = []
    #for i in np.linspace(np.cos(5*u.deg), 1, NProtExp*flat_weight):
        #off_angles['p'].append(np.arccos(i)*u.rad.to(u.deg))
        #weight_p.append(1./flat_weight)
    #off_angles['p'] = np.array(off_angles['p'])
    #print(len(weight_p))
    #print(len(off_angles['p']))



    Rsig = .3
    Rmax = 5
    alpha = 1/(((Rmax/Rsig)**2)-1)
    Non = 0
    Noff = 0
    for s, w, e in zip(chain(gammas['off_angles'], proton['off_angles']),
                       chain(weight_g, weight_p),
                       chain(gammas["MC_Energy"], proton["MC_Energy"])
                       ):
        if .1 < np.log10(e * u.GeV.to(u.TeV)) < .12:
            continue
        if s < Rsig:
            Non += w
        elif s < Rmax:
            Noff += w
    scale = 609
    Non *= 1./scale
    Noff *= 1./scale
    print("Non:", Non)
    print("Noff:", Noff)
    print(alpha)
    print("sigma:", sigma_lima(Non, Noff, alpha=alpha))


    if args.plot:
        plt.style.use('seaborn-talk')
        plt.style.use('t_slides')
        angle_unit = u.deg

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
        plot a sky image of the events '''
        if True:
            fig2 = plt.figure()
            plt.hexbin(
                [(ph-180)*np.sin(th*u.deg) for
                    ph, th in zip(chain(gammas['phi'], proton['phi']),
                                  chain(gammas['theta'], proton['theta']))],
                [a for a in chain(gammas['theta'], proton['theta'])],
                gridsize=41, extent=[-2,2,18,22],
                bins='log'
                )
            plt.colorbar().set_label("log(Number of Events)")
            plt.axes().set_aspect('equal')
            plt.xlabel(r"$\sin(\vartheta) \cdot (\varphi-180) / ${:latex}"
                       .format(angle_unit))
            plt.ylabel(r"$\vartheta$ / {:latex}".format(angle_unit))
            if args.write:
                tikz_save("plots/skymap_{}.tex".format(args.mode))
            #plt.pause(.1)

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
        #plt.hist([off_angles['p']**2,
                  #off_angles['g']**2],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(0, 20),
                 #bins=10)
        #plt.xlabel(r"$\vartheta^2 / \mathrm{"+str(angle_unit)+"}^2$")
        #plt.ylabel("expected events in {}".format(ObsTime))

        #plt.subplot(313)
        plt.hist([np.clip(np.cos(off_angles['p']*u.degree.to(u.rad)), -1, 1-1e-6),
                  np.clip(np.cos(off_angles['g']*u.degree.to(u.rad)), -1, 1-1e-6)],
                 weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 range=(0.996, 1),
                 bins=30)
        plt.xlabel(r"$\cos(\vartheta)$")
        plt.ylabel("expected events in {}".format(ObsTime))
        plt.xlim([0.996, 1])

        plt.tight_layout()

        plt.show()
