from itertools import chain
import astropy.units as u
from scipy.optimize import minimize

import numpy as np

# tau, the superior circle number
np.tau = 2*np.pi

__all__ = ["Sensitivity_PointSource", "crab_source_rate"]


def convert_astropy_array(arr, unit=None):
    """ converts a python list of quantities into a quantified numpy array in the SI unit
    of the same dimension

    Parameters
    ----------
    arr : python list
        list of quantities of same dimension (not strictly exact same unit)
    unit : astropy unit, optional (default: None)
        if set, uses this as the unit of the numpy array
        ought to be of same dimension of the quantities in the list (there is no test)

    Returns
    -------
    a : quantified numpy array
    """

    if unit is None:
        unit = arr[0].unit
        return (np.array([a.to(unit).value for a in arr])*unit).si
    else:
        return np.array([a.to(unit).value for a in arr])*unit


def crab_source_rate(E):
    ''' function for a pseudo-Crab point source rate
    Crab source rate:   dN/dE = 3e-7  * (E/TeV)**-2.48 / (TeV * m² * s)
    (watch out: unbroken power law... not really true)
    norm and spectral index reverse engineered from HESS plot...

    Parameters
    ----------
    E : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    F : astropy quantity
        differential flux at E

    '''
    return 3e-7 * (E/u.TeV)**-2.48 / (u.TeV * u.m**2 * u.s)


def CR_background_rate(E):
    ''' function of the cosmic ray spectrum (simple power law, no knee/ankle)
    Cosmic Ray background rate: dN/dE = 0.215 * (E/TeV)**-8./3 / (TeV * m² * s * sr)
    norm and spectral index reverse engineered from "random" CR plot...

    Parameters
    ----------
    E : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    F : astropy quantity
        differential flux at E

    '''
    return 100 * 0.1**(8./3) * (E/u.TeV)**(-8./3) / (u.TeV * u.m**2 * u.s * u.sr)


def Eminus2(e, unit=u.GeV):
    '''
    boring old E^-2 spectrum

    Parameters
    ----------
    E : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    F : astropy quantity
        differential flux at E

    '''
    return (e/unit)**(-2) / (unit * u.s * u.m**2)


def make_mock_event_rate(spectrum, bin_edges=None, e_min=None, e_max=None,
                         e_unit=u.GeV, n_bins=None, log_e=True, norm=None):
    """
    Creates a histogram with a given binning and fills it according to a spectral function

    Parameters
    ----------

    spectrum : function object
        function of the differential spectrum that shall be sampled into the histogram
        ought to take the energy as an astropy quantity as sole argument
    bin_edges : numpy array, optional (default: None)
        bin edges of the histogram that is to be filled
        either use this or `e_min`, `e_max` and `n_bins` parameters
    e_min, e_max : astropy quantities, optional (defaults: None)
        min and max bin edge of the histogram that is to be filled
        either use these with `n_bins` or `bin_edges` parameter
    e_unit : astropy unit (default: u.GeV)
        unit of the histogram's axis
    n_bins : integer, optional (default: None)
        number of bins of the histogram to be created
        either use this with `e_min` and `e_max` or `bin_edges` parameter
    log_e : bool, optional (default: None)
        tell if the `e_min` and `e_max` parameters are given in logarithm
    norm : float, optional (default: None)
        normalisation factor for the histogram that's being filled

    Returns
    -------
    rates : numpy array
        histogram of the (non-energy-differential) event rate of the proposed spectrum
    bin_edges : numpy array
        bin edges of the histogram -- if `bin_edges` was given in the function call,
        it's the same; if not, it will be the constructed bin edges

    """

    rates = []

    if bin_edges is None:
        if log_e:
            e_min = np.log10(e_min/e_unit)
            e_max = np.log10(e_max/e_unit)
        bin_edges = np.linspace(e_min, e_max, n_bins+1, True)

    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        if log_e:
            bin_centre = 10**((l_edge+h_edge)/2.) * e_unit
            bin_width = (10**h_edge-10**l_edge)*e_unit

        else:
            bin_centre = (l_edge+h_edge) * e_unit / 2.
            bin_width = (h_edge-l_edge)*e_unit
        bin_events = spectrum(bin_centre) * bin_width
        rates.append(bin_events)

    rates = convert_astropy_array(rates)
    if norm:
        rates *= norm/np.sum(rates)

    return rates, bin_edges


def sigma_lima(Non, Noff, alpha=0.2):
    """
    Compute the significance according to Eq. (17) of Li & Ma (1983).

    Parameters
    ----------
    Non : integer
        Number of on counts
    Noff : integer
        Number of off counts
    alpha : float, optional (default: 0.2)
        Ratio of on-to-off exposure

    Returns
    -------
    sigma : float
        the significance of the given off and on counts
    """

    alpha1 = alpha + 1.0
    sum    = Non + Noff
    arg1   = Non / sum
    arg2   = Noff / sum
    term1  = Non  * np.log((alpha1/alpha)*arg1)
    term2  = Noff * np.log(alpha1*arg2)
    sigma  = np.sqrt(2.0 * (term1 + term2))

    return sigma


def diff_to_X_sigma(scale, Non_g, Non_p, Noff_g, Noff_p, alpha, X=5):
    """
    calculates the significance and returns the squared difference to X.
    To be used in a minimiser that determines the necessary source intensity for a
    detection of given significance """

    Non = Non_p + Non_g * scale[0]
    Noff = Noff_p + Noff_g * scale[0]
    sigma = sigma_lima(Non, Noff, alpha)
    return (sigma-X)**2


class Sensitivity_PointSource():
    """
    class to calculate the sensitivity to a known point-source
    """
    def __init__(self, mc_energy_gamma, mc_energy_proton,
                 bin_edges_gamma, bin_edges_proton,
                 energy_unit=u.GeV, flux_unit=u.GeV / (u.m**2*u.s)):
        """
        constructor, simply sets some initial parameters

        Parameters
        ----------
        mc_energy_gamma, mc_energy_proton: quantified numpy arrays
            list of simulated energies of the selected gammas and protons
        bin_edges_gamma, bin_edges_proton : numpy arrays
            list of the bin edges for the various histograms for gammas and protons
            assumes binning in log10(energy)
        energy_unit : astropy quantity, optional (default: u.GeV)
            your favourite energy unit
        flux_unit : astropy quantity, optional (default: u.GeV / (u.m**2 * u.s))
            your favourite differential flux unit

        """

        self.mc_energy_gam = mc_energy_gamma
        self.mc_energy_pro = mc_energy_proton
        self.bin_edges_gam = bin_edges_gamma
        self.bin_edges_pro = bin_edges_proton

        self.energy_unit = energy_unit
        self.flux_unit = flux_unit

    def get_effective_areas(self, n_simulated_gam=None, n_simulated_pro=None,
                            spectrum_gammas=Eminus2, spectrum_proton=Eminus2,
                            Gen_Gammas=None, Gen_Proton=None,
                            total_area_gam=np.tau/2*(1000*u.m)**2,
                            total_area_pro=np.tau/2*(2000*u.m)**2):
        """
        calculates the effective areas for gammas and protons and stores them in the
        class instance

        Parameters
        ----------
        n_simulated_gam, n_simulated_pro : integers, optional (defaults: None)
            number of gamma and proton events used in the MC simulation
            either use this with the `spectrum_gammas` and `spectrum_proton` parameters
            or directly the `Gen_Gammas` and `Gen_Proton` parameters
        spectrum_gammas, spectrum_proton : functions, optional (default: Eminus2)
            function object for the differential generator flux of the gamma and proton
            events
        Gen_Gammas, Gen_Proton : numpy arrays, optional (defaults: None)
            histogram of the generated gammas and protons binned according to
            `.bin_edges_gam` and `.bin_edges_pro`
            either use these directly or generate them with the `n_simulated_...` and
            `spectrum_...` parameters
        total_area_gam, total_area_pro : astropy quantities, optional (defaults:
        pi*(1 km)**2 and pi*(2 km)**2)
            the area within which the shower impact position was generated

        Returns
        -------
        eff_area_gam, eff_area_pro : numpy arrays
            histograms of the effective areas of gammas and protons binned according to
            `.bin_edges_gam` and `.bin_edges_pro`
        """

        if Gen_Gammas is None:
            Gen_Gammas = make_mock_event_rate(
                            spectrum_gammas, norm=n_simulated_gam,
                            bin_edges=self.bin_edges_gam)[0]
        if Gen_Proton is None:
            Gen_Proton = make_mock_event_rate(
                            spectrum_proton, norm=n_simulated_pro,
                            bin_edges=self.bin_edges_pro)[0]

        self.Sel_Gammas = np.histogram(np.log10(self.mc_energy_gam),
                                       bins=self.bin_edges_gam)[0]
        self.Sel_Proton = np.histogram(np.log10(self.mc_energy_pro),
                                       bins=self.bin_edges_pro)[0]

        Efficiency_Gammas = self.Sel_Gammas / Gen_Gammas
        Efficiency_Proton = self.Sel_Proton / Gen_Proton

        self.eff_area_gam = Efficiency_Gammas * total_area_gam
        self.eff_area_pro = Efficiency_Proton * total_area_pro

        return self.eff_area_gam, self.eff_area_pro

    def get_expected_events(self, source_rate=Eminus2, background_rate=CR_background_rate,
                            extension_gamma=None, extension_proton=6*u.deg,
                            observation_time=50*u.h):
        """
        given a source rate and the effective area, calculates the number of expected
        events within a given observation time

        Parameters
        ----------
        source_rate, background_rate : functions, optional (default: `Eminus2` and
            `CR_background_rate`)
            functions for the differential source and background rates
        extension_gamma, extension_proton : astropy quantities, optional (defaults: None
            and 6*u.deg)
            opening angle of the view-cone the events have been generated in
            put `None` for point source
            note: if you use an extension, the flux needs to accomodate that as well
        observation_time : astropy quantity, optional (default: 50*u.h)
            length of the assumed exposure
        """

        # for book keeping
        self.observation_time = observation_time

        SourceRate = make_mock_event_rate(source_rate,
                                          bin_edges=self.bin_edges_gam)[0]
        BackgrRate = make_mock_event_rate(background_rate,
                                          bin_edges=self.bin_edges_pro)[0]

        if extension_gamma:
            omega_gam = np.tau*(1 - np.cos(extension_gamma))*u.rad**2
            SourceRate *= omega_gam
        if extension_proton:
            omega_pro = np.tau*(1 - np.cos(extension_proton))*u.rad**2
            BackgrRate *= omega_pro

        try:
            self.exp_events_per_E_gam = SourceRate * observation_time * self.eff_area_gam
            self.exp_events_per_E_pro = BackgrRate * observation_time * self.eff_area_pro

            return self.exp_events_per_E_gam, self.exp_events_per_E_pro
        except AttributeError as e:
            print("did you call get_effective_areas already?")
            raise e

    def scale_events_to_expected_events(self):
        """
        produces weights to scale the selected events according to source/background
        spectrum and observation time

        Returns
        -------
        weight_g, weight_p : python lists
            weights for the selected gamma and proton events so that they are scaled to
            the number of expected events in `exp_events_per_E_gam` and
            `exp_events_per_E_pro` for every energy bin
        """

        weight_g = []
        weight_p = []
        for ev in self.mc_energy_gam:
            weight_g.append((self.exp_events_per_E_gam/self.Sel_Gammas)[
                                np.digitize(np.log10(ev), self.bin_edges_gam) - 1])
        for ev in self.mc_energy_pro:
            weight_p.append((self.exp_events_per_E_pro/self.Sel_Proton)[
                                np.digitize(np.log10(ev), self.bin_edges_pro) - 1])

        self.weight_g = np.array(weight_g)
        self.weight_p = np.array(weight_p)
        return self.weight_g, self.weight_p

    def get_sensitivity(self, off_angles_g, off_angles_p,
                        min_N=10, max_prot_ratio=.05, Rsig=.3, Rmax=5, verbose=True):

        # the area-ratio of the on- and off-region
        alpha = 1/(((Rmax/Rsig)**2)-1)

        # sensitivities go in here
        sensitivities = []

        for elow, ehigh in zip(10**(self.bin_edges_gam[:-1]),
                               10**(self.bin_edges_gam[1:])):
            Non_g = 0
            Non_p = 0
            Noff_g = 0
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

            scale = minimize(diff_to_X_sigma, [1e-3],
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
            #sensitivities.append([(np.log10(elow)+np.log10(ehigh))/2, sensitivity.value])
            sensitivities.append([np.log10((elow+ehigh)/2.), sensitivity.value])
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
