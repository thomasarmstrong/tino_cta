import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy import units as u

import random

from itertools import chain

from helper_functions import *


def make_mock_generator_spectrum(NEvents, Emin=2, Emax=8, NBins=6):
    '''
    since the energy distribution of the generated events is not accessible,
    this function generates a mock E^-2 spectrum from the number of generated
    shower events '''
    En = []
    we = []

    bin_edges = np.linspace(Emin, Emax, NBins+1)
    for bin in range(NBins):
        En.append((bin_edges[bin] + bin_edges[bin+1])/2.)
        we.append(1./(10**bin_edges[bin]) - 1./(10**bin_edges[bin+1]))

    hist, bin_edges = np.histogram(En, weights=we, bins=NBins,
                                   range=(Emin, Emax), normed=True)
    hist *= NEvents
    return hist, bin_edges





Emin=2
Emax=8
NBins=96
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

    Omega_Proton  = 2*np.pi*(1 - np.cos(6*u.deg))

    NGammas_per_File = 5000 * NReuse_Gammas
    NProton_per_File = 5000 * NReuse_Proton

    NGammas_simulated = NGammas_per_File *  9
    NProton_simulated = NProton_per_File * 51

    Gen_Gammas, edges = make_mock_generator_spectrum(
        NGammas_simulated, Emin=Emin, Emax=Emax, NBins=NBins)
    Gen_Proton, edges = make_mock_generator_spectrum(
        NProton_simulated, Emin=Emin, Emax=Emax, NBins=NBins)

    Sel_Gammas, edges = np.histogram(np.log10(gammas['MC_Energy']),
                                     bins=NBins, range=(Emin, Emax))
    Sel_Proton, edges = np.histogram(np.log10(proton['MC_Energy']),
                                     bins=NBins, range=(Emin, Emax))



    Efficiency_Gammas = Sel_Gammas / Gen_Gammas
    Efficiency_Proton = Sel_Proton / Gen_Proton

    tot_Area_Gammas = np.pi * (1000*u.m)**2
    tot_Area_Proton = np.pi * (2000*u.m)**2

    Eff_Area_Gammas = Efficiency_Gammas * tot_Area_Gammas
    Eff_Area_Proton = Efficiency_Proton * tot_Area_Proton



    print("gammas selected / simulated", NGammas_selected, NGammas_simulated)
    print("proton selected / simulated", NProton_selected, NProton_simulated)


    # Crab source rate:   dN/dE = 3e-7  * (E/TeV)**-2.48 / (TeV * m² * s)
    # CR background rate: dN/dE = 0.215 * (E/TeV)**-.8/3 / (TeV * m² * s * sr)
    # norm and spectral index reverse engineered from HESS and CR plots..
    def crab_source_rate(E):
        return 3e-7 * (E/u.TeV)**-2.48 / (u.TeV * u.m**2 * u.s)
    def CR_background_rate(E):
         100 * 0.1**(8./3) * (E/u.TeV)**-.8/3 / (u.TeV * u.m**2 * u.s * u.sr)
    SourceRate = []
    BackgrRate = []
    SNorm = 3e-7
    BNorm = 100 * 0.1**(8./3)
    for l_edge, h_edge in zip(edges[:-1], edges[1:]):
        bin_centre = 10**((l_edge+h_edge)/2.) * u.GeV
        '''
        differential source rate '''
        bin_value_s = SNorm * (bin_centre/u.TeV)**-2.48
        bin_value_b = BNorm * (bin_centre/u.TeV)**(-8./3.)
        '''
        multiply with bin width '''
        bin_value_s *= ((10**h_edge-10**l_edge)*u.GeV) / u.TeV
        bin_value_b *= ((10**h_edge-10**l_edge)*u.GeV) / u.TeV
        '''
        cosmic ray rate needs to be normalised to solid angle '''
        bin_value_b *= Omega_Proton

        SourceRate.append(bin_value_s)
        BackgrRate.append(bin_value_b)

    SourceRate *= 1./(u.m**2 * u.s)
    BackgrRate *= 1./(u.m**2 * u.s)


    SourceIntensity = 1
    ObsTime = (1*u.h).to(u.s)
    weight_vsE_g = SourceRate * ObsTime * Eff_Area_Gammas * SourceIntensity
    weight_vsE_p = BackgrRate * ObsTime * Eff_Area_Proton

    weight_g = []
    weight_p = []
    for ev in gammas['MC_Energy']:
        weight_g.append(weight_vsE_g[np.digitize(np.log10(ev), edges) - 1])
    for ev in proton['MC_Energy']:
        weight_p.append(weight_vsE_p[np.digitize(np.log10(ev), edges) - 1])

    print("expected gammas:", sum(weight_g))
    print("expected proton:", sum(weight_p))
    NProtExp = sum(weight_p)
    #off_angles['p'] = []
    #flat_weight = 100/NProtExp
    #weight_p = []
    #for i in np.linspace(np.cos(5*u.deg), 1, NProtExp*flat_weight):
        #off_angles['p'].append(np.arccos(i)*u.rad.to(u.deg))
        #weight_p.append(1./flat_weight)
    #off_angles['p'] = np.array(off_angles['p'])
    #print(len(weight_p))
    #print(len(off_angles['p']))

    if args.plot:
        plt.style.use('seaborn-talk')
        plt.style.use('t_slides')
        angle_unit = u.deg

        '''
        plot the source and spectra to be assumed in the sensitivity study '''
        if 0:
            figure = plt.figure()
            plt.subplot(121)
            plt.bar(edges[:-1], SourceRate.value, width=edges[1]-edges[0])
            plt.yscale('log')
            plt.subplot(122)
            plt.bar(edges[:-1], BackgrRate.value, width=edges[1]-edges[0])
            plt.yscale('log')
            plt.pause(.1)

        '''
        plot a sky image of the events '''
        if False:
            fig2 = plt.figure()
            plt.hist2d(
                # convert_astropy_array(chain(gammas['phi'], proton['phi']), unit),
                # convert_astropy_array(chain(gammas['theta'], proton['theta']), unit),
                [a for a in chain(gammas['phi'], proton['phi'])],
                [a for a in chain(gammas['theta'], proton['theta'])],
                range=([[(180-3), (180+3)], [(20-3), (20+3)]]*u.deg) / angle_unit)
            plt.xlabel("phi / {}".format(angle_unit))
            plt.ylabel("theta / {}".format(angle_unit))
            plt.pause(.1)

        '''
        plot the angular distance of the reconstructed shower direction
        from the pseudo source in different scales '''
        figure = plt.figure()
        #plt.subplot(211)
        plt.hist([off_angles['p'], off_angles['g']],
                 weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 range=(0, 5),
                 bins=25)
        plt.xlabel(r"$\alpha / \mathrm{"+str(angle_unit)+"}$")
        plt.ylabel("expected events")
        plt.ylim([0, 3000])

        if args.write:
            tikz_save("plots/"+args.mode+"_proto_significance.tex",
                        draw_rectangles=True)

        #plt.subplot(212)
        #plt.hist([off_angles['p']**2,
                  #off_angles['g']**2],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(0, .75),
                 #bins=25)
        #plt.xlabel(r"$\alpha^2 / \mathrm{"+str(angle_unit)+"}^2$")

        #plt.subplot(313)
        #plt.hist([-np.cos(off_angles['p']*u.degree.to(u.rad)),
                  #-np.cos(off_angles['g']*u.degree.to(u.rad))],
                 #weights=[weight_p, weight_g], rwidth=1, stacked=True,
                 #range=(-1, -.9999),
                 #bins=100)
        #plt.xlabel(r"$-\cos(\alpha)$")

        plt.tight_layout()

        plt.show()
