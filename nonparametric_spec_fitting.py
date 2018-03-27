# ---- Non-Parametric Spectrum fit --------
# This is a parameter file for fitting spectra with a
# non-parametric SFH.  This is set up to use the fractional SFR prior and
# parameterization as described in Leja et al. 2016
# We (optionally) remove the spectral continuum shape by optimizing out a polynomial
# at each model call, if use_continuum is False
# -------------------------------------

import numpy as np
from astropy.table import Table
import os

from sedpy.observate import load_filters
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from astropy.cosmology import WMAP9 as cosmo
from prospect.sources.constants import jansky_cgs, to_cgs_at_10pc
to_cgs = to_cgs_at_10pc

# define paths

components = ['bulge1', 'bulge2', 'bulge3', 'disk', 'clump1', 'clump2', 'clump3']


# --------------
# RUN_PARAMS
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'results/nonpar_spec',
              'infile_spec': 'data/example_spectra_with_noise.txt',
              'infile_phot': 'data/example_mags.txt',
              # dynesty params
              'nested_bound': 'multi',  # bounding method
              #'nested_sample': 'rwalk',  # sampling method
              'nested_sample': 'rslice',  # sampling method
              #'nested_walks': 100,  # MC walks
              'nested_slices': 6,
              'nested_nlive_batch': 100,  # size of live point "batches"
              'nested_nlive_init': 100,  # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0},  # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              'nested_stop_kwargs': {'post_thresh': 0.05, 'n_mc': 50},  # higher threshold, more MCMC
              # Nestle parameters
#              'nestle_npoints': 200,
#              'nestle_method': 'multi',
#              'nestle_maxcall': int(1e7),
              # Data manipulation parameters
              'logify_spectrum': False,
              'normalize_spectrum': False,
              'rescale_spectrum': False,
              'polyorder': 0,
              # SPS parameters
              'i_comp': 0.0,  # spectrum
              'phot': False,  # fit photometry
              'spec': True,  # fit spectrum
              'mask_elines': True,
              'add_neb_emission': False,
              'zred': 2.241,
              'agelims': [0., 7.3, 8.0, 8.5, 9.0, 9.3, 9.5],  # The age bins
              'zcontinuous': 1,
              }

# --------------
# OBS
# --------------

def load_obs(zred=2.241, phot=False, spec=False, mask_elines=False, infile_phot=None, infile_spec=None, i_comp=None, **kwargs):
    """Load a mock

    :param snr:

    :param dlam:

    :returns obs:
        Dictionary of observational data.
    """
    # --- Set component ---
    component = components[int(float(i_comp))-1]

    # --- Set spectrum ---
    if spec:
        table_spec = Table.read(os.getenv('WDIR') + infile_spec, format='ascii')

        # --- Fill the obs dictionary ----
        lumdist = cosmo.luminosity_distance(zred).value
        spec_conversion = (1 + zred) * to_cgs / (lumdist * 1e5)**2 / (3631*jansky_cgs)  # Spectrum in Lsun/Hz per solar mass formed, restframe to observed frame
        obs = {}
        obs['wavelength'] = table_spec['wavelength']*(1.0 + zred)
        obs['spectrum'] = table_spec['spec_' + component] * spec_conversion
        obs['unc'] = table_spec['spec_' + component + '_unc'] * spec_conversion
        obs['mock_snr'] = obs['spectrum']/obs['unc']

        # Masking
        obs['mask'] = np.ones(len(obs['wavelength']), dtype=bool)
        if mask_elines:
            a = (1.0 + zred)  # redshift the mask
            lines = np.array([3729, 3799.0, 3836.5, 3870., 3890.2, 3971.2,  # misc
                              4103., 4341.7, 4862.7, 4960.3, 5008.2,  # hgamma + hdelta + hbeta + oiii
                              5877.2, 5890.0, 6302.1, 6549.9, 6564.6, 6585.3,  # naD + oi + halpha + nii
                              6680.0, 6718.3, 6732.7, 7137.8])  # sii
            obs['mask'] = obs['mask'] & eline_mask(obs['wavelength'], lines * a, 18.0 * a)

    else:
        obs['wavelength'] = None
        obs['spectrum'] = None
        obs['unc'] = None
        obs['mask'] = None

    # --- Set photometry ---
    if phot:
        table_phot = Table.read(os.getenv('WDIR') + infile_phot, format='ascii')
        obs['maggies'] = table_phot['mags_' + component]
        snr = 20.0
        obs['maggies_unc'] = obs['maggies']/snr
        obs['mock_snr'] = snr
        obs['phot_mask'] = np.ones(len(obs['maggies']), dtype=bool)
        filter_folder = os.getenv('WDIR') + 'data/filters/'
        obs['filters'] = load_filters(['acs_wfc_f435w.par', 'acs_wfc_f814w.par', 'wfc3_ir_f110w.par', 'wfc3_ir_f160w.par'], directory=filter_folder)
    else:
        obs['maggies'] = None
        obs['filters'] = None
    return obs


def eline_mask(wave, lines, pad):
    """A little method to apply emission line masks based on wavelength
    intervals like used in Starlight.
    """
    isline = np.zeros(len(wave), dtype=bool)
    for w in lines:
        lo, hi = w-pad, w+pad
        #print(lo, hi)
        isline = isline | ((wave > lo) & (wave < hi))
    return ~isline


# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, add_realism=False, **extras):
    """Load the SPS object.  If add_realism is True, set up to convolve the
    library spectra to an sdss resolution
    """
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def load_gp(**extras):
    return None, None

# -------------
# Transforms
# -------------

def zfrac_to_masses(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions and then to bin mass fractions. The transformation is such that
    sfr fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010

    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions (e.g. Leja 2016)
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    sfr_fraction *= np.array(time_per_bin)
    sfr_fraction /= sfr_fraction.sum()

    masses = total_mass * sfr_fraction
    return masses


def stellar_logzsol(logzsol=0.0, **extras):
    return logzsol


# --------------
# MODEL_PARAMS
# --------------

model_params = []

# --- Distance ---
# This is the redshift.  Because we are not separately supplying a ``lumdist``
# parameter, the distance will be determined from the redshift using a WMAP9
# cosmology, unless the redshift is 0, in which case the distance is assumed to
# be 10pc (i.e. for absolute magnitudes)
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 2.241,
                        'units': '',
                        'prior': priors.TopHat(mini=-0.001, maxi=0.001)})

# --- SFH --------

# This gives the start and stop of each age bin.  We will adjust this in
# load_model() below based on the `agelims` run_param
model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [[0.0, 8.0], [8.0, 8.7]],
                        'units': 'log(yr)'})


# This will be the mass in each bin.  It depends on other free and fixed
# parameters, so it is fixed (see zfrac_to_masses() below).  Alternatively, one
# can sample in this parameter directly.  We will adjust this parameter in
# load_model() below to have the proper size and initial value.
model_params.append({'name': 'mass', 'N': 1,
                     'isfree': False,
                     'depends_on': zfrac_to_masses,
                     'init': 1.,
                     #'prior': priors.LogUniform(mini=1e0, maxi=1e12),
                     'units': r'M$_\odot$',})

# This is the total galaxy mass
model_params.append({'name': 'total_mass', 'N': 1,
                     'isfree': True,
                     'init': 1e10,
                     'units': r'M$_\odot$',
                     'prior': priors.LogUniform(mini=1e10, maxi=1e12)})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

# Auxiliary variable used for sampling sfr_fractions from dirichlet.  We will
# adjust the length and priors for this parameter in load_model() below
model_params.append({'name': 'z_fraction', 'N': 1,
                        'isfree': True,
                        'init': [],
                        'units': None,
                        'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)})

# Since we have zcontinuous > 0 above, the metallicity is controlled by the
# ``logzsol`` parameter.
model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.3,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-1, maxi=0.19)})

# For zcontinuous = 2 this gives the power for the metallicity distribution
# -99 uses a 3-pt smoothing
model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,})

# Explicitly set IMF to Chabrier
model_params.append({'name': 'imf_type', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None})

# --- Dust ---------
# FSPS parameter
# 0 is power-law. 1 is CCM, 2 is Calzetti
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 0,
                        'units': 'index'})

# Young star extra optical depth, see FSPS
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.2,
                        'units': 'optical depth',
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

# Optical depth towards all stars, see FSPS
model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 0.5,
                        'reinit': True,
                        'init_disp': 0.3,
                        'units': 'optical depth',
                        'prior': priors.TopHat(mini=0.0, maxi=2.5)})

# Attenuation curve slope if power-law
model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': -0.7,
                        'init_disp': 0.3,
                        'units': 'power-law slope',
                        'prior': priors.TopHat(mini=-2.5, maxi=-1.3)})

# --- Nebular Emission ------

# FSPS parameter
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': False})

# FSPS parameter
model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'depends_on': stellar_logzsol,
                        'prior': priors.TopHat(mini=-2.0, maxi=0.2)})

# FSPS parameter
model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': True,
                        'init': -2.5,
                        'init_disp': 1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4, maxi=-1)})

# --- Spectral Smoothing ------

# This controls whether we smooth in velocity space, wavelength space, R=lam/dlam
# space (similar to velocity space), or something more complicated.
# Can be one of 'vel' | 'lambda' | 'R' | 'lsf'
model_params.append({'name': 'smoothtype', 'N': 1,
                        'isfree': False,
                        'init': 'vel',
                        'units': 'Smoothing in velocity',
                        })

# This controls the amount of smoothing.  It is always given in terms of a
# dispersion not a FWHM, even when using smoothtype=R (where R is usually
# defined as lambda/FWHM).  The units depend on smoothtype.  See
# prospect.utils.smoothing.smoothspec for details
model_params.append({'name': 'sigma_smooth', 'N': 1,
                        'isfree': True,
                        'init': 200.0,
                        'units': 'km/s',
                        'prior': priors.TopHat(mini=50, maxi=250)})

# You want to have this True, and in fact that is the default.
# FFTs are *much* faster.
model_params.append({'name': 'fftsmooth', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': 'We want to use FFTs to smooth',
                        })

# # This sets the minimum wavelength to consider when smoothing.  It should be
# # somewhat lower than the smallest observed frame wavelength.  Optional, and if
# # not supplied it will be determined from the wavelength vector in the `obs`
# # dictionary.
# model_params.append({'name': 'min_wave_smooth', 'N': 1,
#                         'isfree': False,
#                         'init': 3500.0,
#                         'units': r'$\AA$'})

# # Similar to min_wave_smooth above
# model_params.append({'name': 'max_wave_smooth', 'N': 1,
#                         'isfree': False,
#                         'init': 7800.0,
#                         'units': r'$\AA$'})

# --- Calibration ---------

# What order polynomial?
# set to 0 for no calibration polynomial
model_params.append({'name': 'polyorder', 'N': 1,
                        'isfree': False,
                        'init': 12,
                     })

# Overall normalization of the spectrum. Since we don't have photometry we can't let this float
model_params.append({'name': 'spec_norm', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'units': 'f_true/f_obs',
                        'prior': priors.Normal(mean=1.0, sigma=0.1)})


def load_model(agelims=[], **kwargs):
    """Load the model object.
    """
    # These are parameters that can be set at initialization, e.g. to define the mock.
    fix_params = ['add_neb_emission',
                  'logzsol', 'dust2', 'total_mass',
                  'zred', 'sigma_smooth',
                  'polyorder']

    # Get a handy ordered list of parameter names
    pnames = [p['name'] for p in model_params]

    # set up the age bins and initial zfractions
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    ncomp = len(agelims) - 1
    sfr_fraction = np.ones(ncomp) / ncomp  # constant sfr
    zinit = np.zeros(ncomp - 1)
    zinit[0] = 1 - sfr_fraction[0]
    for i in range(1, len(zinit)):
        zinit[i] = 1.0 - sfr_fraction[i] / np.prod(zinit[:i])

    # THIS IS IMPORTANT
    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr fraction
    alpha = np.arange(ncomp-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    # Adjust SFH model parameters and initial values
    model_params[pnames.index('mass')]['N'] = ncomp
    model_params[pnames.index('agebins')]['N'] = ncomp
    model_params[pnames.index('agebins')]['init'] = agebins
    model_params[pnames.index('z_fraction')]['N'] = len(zinit)
    model_params[pnames.index('z_fraction')]['init'] = zinit
    model_params[pnames.index('z_fraction')]['prior'] = zprior

    # Adjust fixed parameters based on what's in run_params
    for k in fix_params:
        try:
            model_params[pnames.index(k)]['init'] = kwargs[k]
        except(KeyError):
            pass
    if 'zred' in kwargs:
        z = kwargs['zred']
        v = np.array([-100., 100.])  # lower and upper limits of residual velocity offset, km/s
        zlo, zhi = (1 + v/3e5) * (1 + z) - 1.0
        model_params[pnames.index('zred')]['prior'] = priors.TopHat(mini=zlo, maxi=zhi)

    # Instantiate and return model
    return sedmodel.PolySedModel(model_params)
