import numpy as np
from itertools import chain
import astropy.units as u
from scipy.optimize import minimize

def convert_astropy_array(arr, unit=None):
    if unit is None:
        unit = arr[0].unit
        return (np.array([a.to(unit).value for a in arr])*unit).si
    else:
        return np.array([a.to(unit).value for a in arr])*unit

def crab_source_rate(E):
    '''
    Crab source rate:   dN/dE = 3e-7  * (E/TeV)**-2.48 / (TeV * m² * s)
    (watch out: unbroken power law... not really true)
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


def make_mock_event_rate(spectra, bin_edges=None, Emin=None, Emax=None,
                         E_unit=u.GeV, NBins=None, logE=True, norm=None):

    rates = [[] for f in spectra]

    if bin_edges is None:
        if logE:
            Emin = np.log10(Emin/E_unit)
            Emax = np.log10(Emax/E_unit)
        bin_edges = np.linspace(Emin, Emax, NBins+1, True)

    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        if logE:
            bin_centre = 10**((l_edge+h_edge)/2.) * E_unit
            bin_width = (10**h_edge-10**l_edge)*E_unit

        else:
            bin_centre = (l_edge+h_edge) * E_unit / 2.
            bin_width = (h_edge-l_edge)*E_unit
        for i, spectrum in enumerate(spectra):
            bin_events = spectrum(bin_centre) * bin_width
            rates[i].append(bin_events)

    for i, rate in enumerate(rates):
        rate = convert_astropy_array(rate)
        if norm:
            rate *= norm[i]/np.sum(rate)
        rates[i] = rate

    return (*rates), bin_edges


def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute Eq. (17) of Li & Ma (1983).

    Parameters:
     Non   - Number of on counts
     Noff  - Number of off counts
    Keywords:
     alpha - Ratio of on-to-off exposure
    """

    alpha1 = alpha + 1.0
    sum    = Non + Noff
    arg1   = Non / sum
    arg2   = Noff / sum
    term1  = Non  * np.log((alpha1/alpha)*arg1)
    term2  = Noff * np.log(alpha1*arg2)
    sigma  = np.sqrt(2.0 * (term1 + term2))

    return sigma


def diff_to_5_sigma(scale, Non_g, Non_p, Noff_g, Noff_p, alpha):
    """
    calculates the significance and returns the squared difference to 5.
    To be used in a minimiser that determines the necessary source intensity for a
    five sigma detection """

    Non = Non_p + Non_g * scale[0]
    Noff = Noff_p + Noff_g * scale[0]
    sigma = sigma_lima(Non, Noff, alpha)
    return (sigma-5)**2


class Sensitivity_PointSource():
    def __init__(self, mc_energy_gamma, mc_energy_proton,
                 bin_edges_gamma, bin_edges_proton,
                 energy_unit=u.GeV, flux_unit=u.erg / (u.m**2*u.s)):

        self.mc_energy_gam = mc_energy_gamma
        self.mc_energy_pro = mc_energy_proton
        self.bin_edges_gam = bin_edges_gamma
        self.bin_edges_pro = bin_edges_proton

        self.energy_unit = energy_unit
        self.flux_unit = flux_unit

    def get_effective_areas(self, n_simulated_gam, n_simulated_pro,
                            spectrum_gammas=Eminus2, spectrum_proton=Eminus2,
                            total_area_gam=np.pi*(1000*u.m)**2,
                            total_area_pro=np.pi*(2000*u.m)**2):

        Gen_Gammas = make_mock_event_rate(
                        [spectrum_gammas], norm=[n_simulated_gam],
                        bin_edges=self.bin_edges_gam)[0]
        Gen_Proton = make_mock_event_rate(
                        [spectrum_proton], norm=[n_simulated_pro],
                        bin_edges=self.bin_edges_pro)[0]

        Sel_Gammas = np.histogram(np.log10(self.mc_energy_gam),
                                  bins=self.bin_edges_gam)[0]
        Sel_Proton = np.histogram(np.log10(self.mc_energy_pro),
                                  bins=self.bin_edges_pro)[0]

        Efficiency_Gammas = Sel_Gammas / Gen_Gammas
        Efficiency_Proton = Sel_Proton / Gen_Proton

        self.eff_area_gam = Efficiency_Gammas * total_area_gam
        self.eff_area_pro = Efficiency_Proton * total_area_pro

        return self.eff_area_gam, self.eff_area_pro

    def get_expected_events(self, source_rate=Eminus2, background_rate=CR_background_rate,
                            extension_gamma=None, extension_proton=6*u.deg,
                            observation_time=50*u.h):
        # for book keeping
        self.observation_time = observation_time

        SourceRate = make_mock_event_rate([source_rate],
                                          bin_edges=self.bin_edges_gam)[0]
        BackgrRate = make_mock_event_rate([background_rate],
                                          bin_edges=self.bin_edges_pro)[0]

        if extension_gamma:
            omega_gam = 2*np.pi*(1 - np.cos(extension_gamma))*u.rad**2
            SourceRate *= omega_gam
        if extension_proton:
            omega_pro = 2*np.pi*(1 - np.cos(extension_proton))*u.rad**2
            BackgrRate *= omega_pro

        try:
            self.exp_events_per_E_gam = SourceRate * observation_time * self.eff_area_gam
            self.exp_events_per_E_pro = BackgrRate * observation_time * self.eff_area_pro

            return self.exp_events_per_E_gam, self.exp_events_per_E_pro
        except AttributeError:
            print("did you call get_effective_areas already?")
            return [np.nan], [np.nan]

    def scale_events_to_expected_events(self):

        weight_g = []
        weight_p = []
        for ev in self.mc_energy_gam:
            weight_g.append(self.exp_events_per_E_gam[
                                np.digitize(np.log10(ev), self.bin_edges_gam) - 1])
        for ev in self.mc_energy_pro:
            weight_p.append(self.exp_events_per_E_pro[
                                np.digitize(np.log10(ev), self.bin_edges_pro) - 1])

        self.weight_g = np.array(weight_g) / np.sum(weight_g) \
            * np.sum(self.exp_events_per_E_gam)
        self.weight_p = np.array(weight_p) / np.sum(weight_p) \
            * np.sum(self.exp_events_per_E_pro)

        return self.weight_g, self.weight_p

    def get_sensitivity(self, off_angles_g, off_angles_p,
                        min_N=10, max_prot_ratio=.05, Rsig=.3, Rmax=5, verbose=True):

        ''' the area-ratio of the on- and off-region '''
        alpha = 1/(((Rmax/Rsig)**2)-1)

        ''' sensitivities go in here '''
        sensitivities = []

        for elow, ehigh in zip(10**(self.bin_edges_gam[:-1]),
                               10**(self.bin_edges_gam[1:])):
            Non_g = 0
            Noff_g = 0
            Non_p = 0
            Noff_p = 0

            for s, w in zip(chain(off_angles_g[(self.mc_energy_gam > elow) &
                                               (self.mc_energy_gam < ehigh)],
                                  off_angles_p[(self.mc_energy_pro > elow) &
                                               (self.mc_energy_pro < ehigh)]),
                            chain(self.weight_g[(self.mc_energy_gam > elow) &
                                                (self.mc_energy_gam < ehigh)],
                                  self.weight_p[(self.mc_energy_pro > elow) &
                                               (self.mc_energy_pro < ehigh)])
                            ):
                if s < Rsig:
                    Non_g += w
                elif s < Rmax:
                    Noff_g += w

            if Non_g == 0:
                continue

            scale = minimize(diff_to_5_sigma, [1e-3],
                            args=(Non_g, Non_p, Noff_g, Noff_p, alpha),
                            # method='BFGS',
                            method='L-BFGS-B', bounds=[(1e-4, None)],
                            options={'disp': False}
                            ).x[0]

            if verbose:
                print("e low {}\te high {}".format(np.log10(elow),
                                                   np.log10(ehigh)))

            Non_g *= scale
            Noff_g *= scale

            scale_a = check_min_N(Non_g, Noff_g, Non_p, Noff_p, scale,
                                  min_N, verbose)
            Non_g *= scale_a
            Noff_g *= scale_a
            scale *= scale_a


            scale_r = check_background_contamination(Non_g, Noff_g, Non_p, Noff_p, scale,
                                           max_prot_ratio, verbose)

            Non_g *= scale_r
            Noff_g *= scale_r
            scale *= scale_r

            flux = Eminus2((elow+ehigh)/2.).to(self.flux_unit)
            sensitivity = flux*scale
            sensitivities.append([(np.log10(elow)+np.log10(ehigh))/2, sensitivity.value])
            if verbose:
                print("sensitivity: ", sensitivity)
                print("Non:", Non_g+Non_p)
                print("Noff:", Noff_g+Noff_p)
                print("  {}, {}, {}, {}".format(Non_g, Noff_g, Non_p, Noff_p))
                print("alpha:", alpha)
                print("sigma:", sigma_lima(Non_g+Non_p, Noff_g+Noff_p, alpha=alpha))

                print()

        return sensitivities


def check_min_N(Non_g, Noff_g, Non_p, Noff_p, scale, min_N, verbose=True):
    if Non_g+Noff_g+Non_p+Noff_p < min_N:
        scale_a = (min_N-Non_p-Noff_p) / (Non_g+Noff_g)

        if verbose:
            print("  N_tot too small: {}, {}, {}, {}".format(Non_g, Noff_g,
                                                             Non_p, Noff_p))
            print("  scale_a:", scale_a)

        return scale_a
    else:
        return 1


def check_background_contamination(Non_g, Noff_g, Non_p, Noff_p, scale,
                                   max_prot_ratio, verbose=True):
    if Non_p / (Non_g+Non_p) > max_prot_ratio:
        scale_r = (1-max_prot_ratio) * Non_p / Non_g
        if verbose:
            print("  too high proton contamination: {}, {}".format(Non_g, Non_p))
            print("  scale_r:", scale_r)
        return scale_r
    else:
        return 1
