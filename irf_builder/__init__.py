import numpy as np
from astropy import units as u

from .irfs import effective_areas

from .meta_data_loader import load_meta_data_from_yml as load_meta_data

from . import spectra

from .weighting import unbinned_wrapper as make_weights

from .event_selection import minimise_sensitivity_per_bin as optimise_cuts

from .sensitivity import point_source_sensitivity as calculate_sensitivity

mc_energy_name = "MC_Energy"
reco_energy_name = "reco_Energy"

# your favourite units here
angle_unit = u.deg
energy_unit = u.TeV
flux_unit = (u.erg * u.cm**2 * u.s)**(-1)
sensitivity_unit = flux_unit * u.erg**2

# define edges to sort events in
e_bin_edges = np.logspace(-2, 2.5, 20) * u.TeV
e_bin_centres = np.sqrt(e_bin_edges[:-1] * e_bin_edges[1:])
e_bin_edges_fine = np.logspace(-2, 2.5, 100) * u.TeV
e_bin_centres_fine = np.sqrt(e_bin_edges_fine[:-1] * e_bin_edges_fine[1:])


# use `meta_data_loader` to put information concerning the MC production in here
meta_data = {"units": {}, "gamma": {}, "proton": {}, "electron": {}}
