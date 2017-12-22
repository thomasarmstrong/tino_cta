import numpy as np
from astropy import units as u
import scipy.integrate as integrate


__all__ = ["crab_source_rate",
           "cr_background_rate",
           "electron_spectrum"]


def crab_source_rate(energy):
    '''
    function for a pseudo-Crab point-source rate:
        dN/dE = 3e-7  * (E/TeV)**-2.5 / (TeV * m² * s)
    (watch out: unbroken power law... not really true)
    norm and spectral index reverse engineered from HESS plot...

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return 3e-7 * (energy / u.TeV)**-2.5 / (u.TeV * u.m**2 * u.s)


def cr_background_rate(energy):
    '''
    function for the Cosmic Ray background rate:
        dN/dE = 0.215 * (E/TeV)**-8./3 / (TeV * m² * s * sr)
    (simple power law, no knee/ankle)
    norm and spectral index reverse engineered from "random" CR plot...

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return 100 * 0.1**(8. / 3) * (energy / u.TeV)**(-8. / 3) / \
        (u.TeV * u.m**2 * u.s * u.sr)


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


def e_minus_2(energy, unit=u.TeV):
    '''
    boring, old, unnormalised E^-2 spectrum

    Parameters
    ----------
    energy : astropy quantity
        energy for which the rate is desired

    Returns
    -------
    flux : astropy quantity
        differential flux at E

    '''
    return (energy / unit)**(-2) / (unit * u.s * u.m**2)


def make_mock_event_rate(spectrum, bin_edges, log_e=False, norm=None):
    """
    Creates a histogram with a given binning and fills it according to a spectral function

    Parameters
    ----------
    spectrum : function object
        function of the differential spectrum that shall be sampled into the histogram
        ought to take the energy as an astropy quantity as sole argument
    bin_edges : numpy array, optional (default: None)
        bin edges of the histogram that is to be filled
    log_e : bool, optional (default: False)
        tell if the values in `bin_edges` are given in logarithm
    norm : float, optional (default: None)
        normalisation factor for the histogram that's being filled
        sum of all elements in the array will be equal to `norm`

    Returns
    -------
    rates : numpy array
        histogram of the (non-energy-differential) event rate of the proposed spectrum
    """

    def spectrum_value(e):
        """
        `scipy.integrate` does not like units during integration. use this as a quick fix
        """
        return spectrum(e).value

    rates = []
    if log_e:
        bin_edges = 10**bin_edges
    for l_edge, h_edge in zip(bin_edges[:-1], bin_edges[1:]):
        bin_events = integrate.quad(
            spectrum_value, l_edge.value, h_edge.value)[0]
        rates.append(bin_events)

    # units have been strip for the integration. the unit of the result is the unit of the
    # function: spectrum(e) times the unit of the integrant: e -- for the latter use the
    # first entry in `bin_edges`
    rates = np.array(rates) * spectrum(bin_edges[0]).unit * bin_edges[0].unit

    # if `norm` is given renormalise the sum of the `rates`-bins to this value
    if norm:
        rates *= norm / np.sum(rates)

    return rates
