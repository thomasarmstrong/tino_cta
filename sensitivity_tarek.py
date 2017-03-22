import numpy as np

from astropy.table import Table
from astropy import units as u

from itertools import chain

from helper_functions import *

# imports the PS module
from ctapipe.analysis.Sensitivity import *
# imports a few spectral flux functions
from ctapipe.analysis.Sensitivity import (crab_source_rate, Eminus2, CR_background_rate,
                                          )

# FIXME need to know:
# • N simulated events
# • simulated energy spectrum
# • generator areas


if __name__ == "__main__":

    edges_energy = np.linspace(0, 6, 28)

    gammas = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_gamma_3_3HB1-ND.fits")
    proton = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_proton_3_3HB1-ND.fits")
    electr = Table.read("/local/home/tmichael/Data/cta/Tarek/"
                        "CTA_MARS_electron_3_3HB1-ND.fits")

    print("E min, max gammas:", np.min(gammas['MC_ENERGY']), np.max(gammas['MC_ENERGY']))
    print("E min, max proton:", np.min(proton['MC_ENERGY']), np.max(proton['MC_ENERGY']))
    print("E min, max electr:", np.min(electr['MC_ENERGY']), np.max(electr['MC_ENERGY']))

    gen_spectrum = lambda x: (x/u.GeV) ** (-2) / (u.GeV * u.s * u.m**2)

    NGammas_simulated = 5e6
    Nproton_simulated = 5e6
    Nelectr_simulated = 5e6

    gen_area_g = (5*u.km)**2 * np.pi
    gen_area_p = (5*u.km)**2 * np.pi
    gen_area_g = (5*u.km)**2 * np.pi

    mask_gammas = (gammas['MARS_ALT'] != -1) & (gammas['MARS_AZ'] != -1)
    mask_proton = (proton['MARS_ALT'] != -1) & (proton['MARS_AZ'] != -1)
    mask_electr = (electr['MARS_ALT'] != -1) & (electr['MARS_AZ'] != -1)

    gammas_m = gammas[mask_gammas]
    proton_m = proton[mask_proton]
    electr_m = electr[mask_electr]

    off_angles_gammas = np.sqrt((gammas_m['MARS_AZ']*np.cos(gammas_m['MARS_ALT']))**2 +
                                gammas_m['MARS_ALT']**2)
    off_angles_proton = np.sqrt((proton_m['MARS_AZ']*np.cos(proton_m['MARS_ALT']))**2 +
                                proton_m['MARS_ALT']**2)
    off_angles_electr = np.sqrt((electr_m['MARS_AZ']*np.cos(electr_m['MARS_ALT']))**2 +
                                electr_m['MARS_ALT']**2)

    print("setting up 'Sensitivity_PointSource' object")
    Sens = Sensitivity_PointSource(
        mc_energies={'g': gammas_m['MC_ENERGY'], 'p': proton_m['MC_ENERGY'],
                     'e': electr_m['MC_ENERGY']},
        off_angles={'g': off_angles_gammas, 'p': off_angles_proton,
                    'e': off_angles_electr},
        energy_bin_edges={'g': edges_energy, "p": edges_energy, 'e': edges_energy},
        energy_unit=u.GeV, flux_unit=u.erg/(u.m**2*u.s))
    print("... done")

    print("calling 'calculate_sensitivities'")
    sensitivities = Sens.calculate_sensitivities(
                    n_simulated_events={'g': NGammas_simulated,
                                        'p': Nproton_simulated,
                                        'e': Nelectr_simulated},

                    generator_spectra={'g': gen_spectrum,
                                       'p': gen_spectrum,
                                       'e': gen_spectrum},

                    rates={'g': Eminus2, 'p': CR_background_rate, 'e': Eminus2},
                    generator_areas={'g': gen_area_g,
                                     'p': gen_area_p,
                                     'e': gen_area_g})
    print("... done")



    plt.figure()
    plt.loglog(
        10 ** edges_energy[:-1],
        Sens.effective_areas['g'])
    plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
    plt.ylabel("effective area")
    plt.suptitle("gammas")

    plt.figure()
    plt.loglog(
        sensitivities["Energy MC"],
        sensitivities["Sensitivity"])
    plt.xlabel(r"$E_\mathrm{MC} / \mathrm{GeV}$")
    plt.ylabel("sensitivity")
    plt.show()
