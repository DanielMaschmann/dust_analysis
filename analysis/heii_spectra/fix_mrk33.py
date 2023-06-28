import os , glob
import numpy as np
import pandas as pd
np.seterr(all='ignore')  # hides irrelevant warnings about divide-by-zero, etc


import astropy.units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling import fitting
from astropy.nddata import StdDevUncertainty

from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.fitting import fit_lines
from specutils.fitting import fit_continuum
from specutils.manipulation import extract_region

from dust_extinction.parameter_averages import CCM89

import matplotlib.pyplot as plt

from specutils.manipulation import gaussian_smooth


path = '/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec'
meta_df = pd.read_csv('/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/s1x_id_databaseA.csv'
                      ).set_index(['target_name', 'filters'])
targets = [tmp[0] for tmp in meta_df.index.to_numpy()[::3]]


# add Av to df
Av_dict = {'HE_2_10': 0.306,
           'NGC_3049': 0.105,
           'NGC_3125': 0.21,
           'MRK_33': 0.033,
           'NGC_4214': 0.06,
           'NGC_4670': 0.041,
           'TOL_89': 0.181,
           'TOL_1924_416': 0.236}

meta_df['Av'] = 0
meta_df['objID'] = ''

for i,row in meta_df.iterrows():
    t,f = i
    Av = Av_dict[t]
    meta_df.loc[i, 'Av'] = Av_dict[t]
#     meta_df.loc[i,'objID'] = row.file_id[:-9]

    if (f=='G140L') & np.isin(t, ['HE_2_10', 'NGC_3049', 'NGC_3125', 'TOL_89', 'TOL_1924_416']):
        meta_df.loc[i, 'prefix'] = 'combined'


def smooth(array, npix=4):

    n=len(array)
    medians=[np.median(array[i:i+npix]) for i in range(0,n,npix)]

    return np.array(medians)


## optical fitting windows
span = [4000, 5500]
emlines=pd.read_csv('/home/benutzer/Documents/projects/dust_analysis/uv spectral analysis/scripts/emlins.csv')
opt_cont_bad = np.array([(2850, 4000)]+[(tmp-5, tmp+5) for tmp in emlines.wavelength if (tmp > 4000) and (tmp <5500)] + [(5500, 5700)])

#### UV Continuum
calzetti_windows= np.array([(1265,1285),
                            (1305,1320),
                            (1345,1370),
                            (1415,1490),
                            (1565,1585),
                            (1625, 1630),
                            (1658, 1673),
                            (1695,1700)])

log_calzetti_windows = np.log10(calzetti_windows)
#### Windows to fit uv continuum
## Table 2 Calzetti et al 1994
## https://articles.adsabs.harvard.edu/pdf/1994ApJ...429..582C

## optical fitting windows
emlines=pd.read_csv('/home/benutzer/Documents/projects/dust_analysis/heII/scripts/final_reduction_scripts/optical_emlines.csv')
emline_windows = np.array((emlines.Line.values-10, emlines.Line.values+10)).T

u_flux = u.erg/u.s/u.cm/u.cm/u.AA
u_xlam = u.AA

def _log_spectra(flux, xlam, err):

    flux=np.where(flux>0, flux, np.nan) # negative numbers make errors in log-space

    ##
    log_flux=np.log10(flux)
    log_xlam=np.log10(xlam)
    log_err=err/(flux*np.log(10))

    return Spectrum1D(flux=log_flux*u_flux, spectral_axis=log_xlam*u_xlam, uncertainty=StdDevUncertainty(log_err, u_flux))


def _fit_continuum(flux, xlam, err, grating):

    if grating == 'G140L':

        log_spec = _log_spectra(flux, xlam, err)
        log_sub_spec = extract_region(log_spec, SpectralRegion(log_calzetti_windows * u.AA), return_single_spectrum=True)

        continuum_model=fit_continuum(log_sub_spec,
                                      models.Linear1D(),
                                      fitter=fitting.LinearLSQFitter(calc_uncertainties=True))
        continuum_flux= 10**continuum_model(log_spec.spectral_axis).value * u_flux

    else:
        #spec = Spectrum1D(flux = flux*u_flux, spectral_axis = xlam * u_xlam, uncertainty = StdDevUncertainty(err, u_flux))
        #sub_spec = extract_region(spec, SpectralRegion(4000*u.AA, 5500*u.AA), return_single_spectrum=True)

        log_spec = _log_spectra(flux, xlam, err)
        log_sub_spec = extract_region(log_spec, SpectralRegion(np.log10(4000)*u.AA, np.log10(5500)*u.AA), return_single_spectrum=True)

        continuum_model = fit_continuum(log_sub_spec,
                                        models.Linear1D(),
                                        fitter=fitting.LinearLSQFitter(calc_uncertainties=True),
                                        exclude_regions=SpectralRegion(np.log10(emline_windows) * u.AA))
#         continuum_flux = continuum_model(spec.spectral_axis)
        continuum_flux= 10**continuum_model(log_spec.spectral_axis).value * u_flux


    return continuum_model, continuum_flux

lines={'heII_1640': {'mean':1640*u.AA, 'fit_window':[1631*u.AA, 1653*u.AA]},
       'heII_4686': {'mean':4686*u.AA, 'fit_window':[4670*u.AA, 4706*u.AA]},
       'hbeta': {'mean':4861*u.AA, 'fit_window':[4847*u.AA, 4868*u.AA]},
       'hgamma':{'mean':4341*u.AA, 'fit_window':[4329*u.AA, 4353*u.AA]}
      }


def _integrate_flux(spec1d_sub, emission_line):

    region=SpectralRegion(*lines[emission_line]['fit_window'])
    sub_spec=extract_region(spec1d_sub, region)

    ## Calculate line flux
    flux=sub_spec.flux
    dx = np.abs(np.diff(sub_spec.spectral_axis.bin_edges))
    line_flux = np.sum(flux * dx)

    ## Calculate the error
    variance_q = sub_spec.uncertainty.quantity ** 2
    line_flux.uncertainty = np.sqrt(np.sum(variance_q * dx**2))

    return (line_flux.value, line_flux.uncertainty.value)

def _fit_emission(_spec, band, target):

    print('band ', band)
    print('target ', target)

    if band=='G140L':
        if target!='MRK_33':
            g=[models.Gaussian1D(mean=1640*u.AA, amplitude=2e-15*u_flux)]
            kwargs={'window':10*u.AA,
                    'exclude_regions':[SpectralRegion(1500*u.AA, 1590*u.AA)]}
        else:
            g=[models.Gaussian1D(mean=1640*u.AA, amplitude=2e-15*u_flux, stddev=5*u.AA)]
            kwargs={'window':15*u.AA,
                    'exclude_regions':[SpectralRegion(1500*u.AA, 1590*u.AA)]}

    elif band=='G430L':
        g=[models.Gaussian1D(mean=ln, stddev=8*u.AA) for ln in [4341*u.AA, 4686*u.AA, 4861*u.AA]]
        kwargs={'window':20*u.AA,
                'exclude_regions':[SpectralRegion(4500*u.AA, 4570*u.AA)]}

    else:

        if target=='HE_2_10':
            g=[models.Gaussian1D(mean=4686*u.AA, stddev=2*u.AA)]#4*u.AA)]
            kwargs={'window':(4679*u.AA, 4708*u.AA), 'exclude_regions':None}

        elif target=='NGC_3049':
            g=[models.Gaussian1D(mean=4692*u.AA, stddev=.65*u.AA)]
            kwargs={'window':(4679*u.AA, 4708*u.AA), 'exclude_regions':None}


        elif target=='NGC_4214':
            g=[models.Gaussian1D(mean=4686*u.AA, stddev=1.5*u.AA)]
            kwargs={'window':(4676*u.AA,4696*u.AA) , 'exclude_regions':None}

        elif target=='TOL_89':
            g=[models.Gaussian1D(mean=4686*u.AA, stddev=1.5*u.AA)]
            kwargs={'window':(4676*u.AA,4696*u.AA) , 'exclude_regions':None}

        else:
            g=[models.Gaussian1D(mean=4686*u.AA, stddev=3.5*u.AA)]#4*u.AA)]
            kwargs={'window':10*u.AA, 'exclude_regions':None}

    line_fit=fit_lines(_spec, g, fitter=fitting.LevMarLSQFitter(calc_uncertainties=True), **kwargs)
    dx=np.abs(np.diff(_spec.spectral_axis.bin_edges))

    fitted_flux=[tmp(_spec.spectral_axis) for tmp in line_fit]
    line_flux=[np.sum(tmp*dx) for tmp in fitted_flux]

    return line_fit, fitted_flux, line_flux


class _specData:
    def __init__(self, target, band, tr='A'):

        self.target = target
        self.band = band

        targ_data = meta_df.loc[(target, band)].copy()
        self.targ_data=targ_data
        self.file = f'/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/{target}/{targ_data.prefix}{tr}_sx1.fits'
        self.z = targ_data['z']
        self.Av = targ_data['Av']

        tab = fits.getdata(self.file, 1)
        self.obs_xlam = tab[0]['WAVELENGTH']*u_xlam
        self.obs_flux = tab[0]['FLUX']*u_flux
        self.err = StdDevUncertainty(tab[0]['ERROR']/(1+self.z),u_flux)

        self.flux = self.mw_de_redden_flux(self.obs_flux, self.obs_xlam).value/(1+self.z) * u_flux
        self.xlam = self.obs_xlam.value/(1+self.z) * u_xlam
        self.spec1d = Spectrum1D(flux=self.flux, spectral_axis=self.xlam, uncertainty=self.err)

        print('observed line ', 4686 * (1+self.z))

        if band == 'G430L':
            npix = 2
        elif band == 'G430M':
            npix = 3
            self.xlam = smooth(self.xlam.value, npix=npix)*u_xlam
            self.flux = smooth(self.flux.value, npix=npix)*u_flux
            self.err = StdDevUncertainty(smooth(self.err.array, npix=npix),u_flux)
            self.spec1d = Spectrum1D(flux=self.flux,
                                   spectral_axis=self.xlam,
                                   uncertainty=self.err)
            npix = 1

        else:
            npix = 3
        tmp_sp = Spectrum1D(flux=self.flux, spectral_axis=self.xlam, uncertainty=self.err)
        tmp_spec = gaussian_smooth(tmp_sp, stddev=npix)
        x, flux, err = tmp_spec.spectral_axis.value, tmp_spec.flux.value, tmp_spec.uncertainty.array

        self.xlam = x * u_xlam
        self.flux = flux * u_flux
        self.err = StdDevUncertainty(err, u_flux)
        self.spec1d = Spectrum1D(flux=self.flux,
                               spectral_axis=self.xlam,
                               uncertainty=self.err)

        # ------- fit continuum
        if band == 'G140L':
            self.continuum_fitting_window = calzetti_windows
        else:
            self.continuum_fitting_window = opt_cont_bad

        self.continuum_model, self.continuum_flux = _fit_continuum(self.flux.value, self.xlam.value, self.err.array, band)
        self.spec1d_sub = self.spec1d-self.continuum_flux

        # ------- Fitting lines
        model, fitted_flux, line_flux = _fit_emission(self.spec1d_sub, self.band, self.target)
        self.line_model = model
        self.fitted_flux = fitted_flux
        self.line_flux_fit = line_flux

    # Line Flux from spectra
        self.line_flux_spec = {'f1640': None, 'f4686': None, 'hb': None, 'hg': None}
        self.line_flux_spec_err = {'f1640': None, 'f4686': None, 'hb': None, 'hg': None}
        self._get_flux_from_spec()

    def mw_de_redden_flux(self, obs_flux, obs_xlam):
        # Correct for interstellar reddening from milky way using Mathis 1990 law
        # xlam -- Angstrom (observed wavelength NOT restframe)
        # flux -- erg/s/cm/cm/Angstrom
        Av = self.Av
        ext = CCM89(Rv=3.1)
        dust = ext.extinguish(obs_xlam, Av=Av)

        return obs_flux/dust

    def _get_flux_from_spec(self):

        emission_lines = {'f1640': {'mean': 1640, 'fit_window': [1630, 1652]},
                          'f4686': {'mean': 4686, 'fit_window': [4670, 4705]},
                          'hb': {'mean': 4861, 'fit_window': [4847, 4868]},
                          'hg': {'mean': 4341, 'fit_window': [4329, 4353]}
                          }

        X = self.spec1d.spectral_axis.value
        Y = (self.spec1d.flux.value - self.continuum_flux.value)/1e-15
        E = self.err.array/1e-15

        for k in emission_lines.keys():
            ln = emission_lines[k]['mean']

            if (X[-1] >= ln) and (X[0] <= ln):

                fit_window = emission_lines[k]['fit_window']
                if self.band == 'G430M':
                    sel_sum = np.where(((X >= fit_window[0]) & (X <= 4676) |
                                        (X >= 4678.5) & (X <= fit_window[1])))[0]
                else:
                    sel_sum = np.where((X >= fit_window[0]) & (X <= fit_window[1]))[0]

                fit_sum_pars = {"X": X[sel_sum],
                                "Y": Y[sel_sum],
                                "E": E[sel_sum]}

                nsim = 1000
                line_sum_all = []
                for ss in range(nsim):
                    xx = fit_sum_pars["X"]
                    yy = np.random.normal(fit_sum_pars["Y"], fit_sum_pars["E"], len(fit_sum_pars["X"]))
                    line_sum_all.append(np.trapz(yy, xx))

                tmp = np.percentile(line_sum_all, [50-68/2,50,50+68/2])
                line_sum_med = tmp[1]
                line_sum_up = tmp[2]
                line_sum_low = tmp[0]

                self.line_flux_spec[k] = np.round(line_sum_med,6)
                self.line_flux_spec_err[k] = [np.round((line_sum_med-line_sum_low),6), np.round((line_sum_up-line_sum_med),6)]


bands = ['G140L', 'G430L', 'G430M']

all_spec = {}


if glob.glob(f'/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/MRK_33/*B_sx1.fits'):
    print('MRK_33' + 'B')
    spec = {b: _specData('MRK_33', b, tr='B') for b in bands}
    all_spec['MRK_33' + 'B'] = spec


lines = {'[He II] 1640': 1640*u.Angstrom,
         'hgamma': 4341*u.Angstrom,
         '[He II] 4686': 4686*u.Angstrom,
         'hbeta': 4861*u.Angstrom}

INFO = {
    'G140L': {
        "lines": [1640*u.AA],
        "fit_window": [[1630*u.AA, 1652*u.AA]],
        "xlimit": (1550,1700)},

    'G430L': {
        "lines": [4341*u.AA, 4686*u.AA, 4861*u.AA],
        "fit_window": [4341*u.AA + np.array([-30,30])*u.AA,
                       [4670*u.AA, 4705*u.AA],
                        4861*u.AA + np.array([-30,30])*u.AA],
        "xlimit": (4310, 4905)},

    'G430M': {
        "lines": [4686*u.AA],
        "fit_window": [[4670*u.AA, 4705*u.AA]],
        "xlimit": (4580,4820)}}


plot_dict = {0: {'b': 'G140L', 'line': 'f1640', 'window': np.array([[1630, 1652]])*u.AA, 'xlimit': (1550, 1700)},
             1: {'b': 'G430M', 'line': 'f4686', 'window': np.array([[4670, 4705]])*u.AA, 'xlimit': (4590, 4810)},
             2: {'b': 'G430L', 'line': 'hg', 'window': np.array([[4332, 4353], [4670, 4705], [4849, 4870]])*u.AA, 'xlimit': (4200, 4470)},
             3: {'b': 'G430L', 'line': 'hb', 'window': np.array([[4332, 4353], [4670, 4705], [4849, 4870]])*u.AA, 'xlimit': (4650, 4950)}}


# save all targets
for target_id in ['MRK_33B']:

    spec_data_g140l = all_spec[target_id]['G140L']
    spec_data_g430m = all_spec[target_id]['G430M']

    continuum_g140l = spec_data_g140l.continuum_flux
    spec_g140l = spec_data_g140l.spec1d
    wave_g140l = spec_g140l.spectral_axis
    flux_g140l = spec_g140l.flux
    flux_err_g140l = spec_data_g140l.err.array

    continuum_g430m = spec_data_g430m.continuum_flux
    spec_g430m = spec_data_g430m.spec1d
    wave_g430m = spec_g430m.spectral_axis
    flux_g430m = spec_g430m.flux
    flux_err_g430m = spec_data_g430m.err.array

    # fitting results
    beta = spec_data_g140l.continuum_model.slope.value
    beta_err = spec_data_g140l.continuum_model.stds['slope']
    line_flux_f1640 = spec_data_g140l.line_flux_spec['f1640']
    line_flux_err_f1640 = spec_data_g140l.line_flux_spec_err['f1640']
    line_flux_f4686 = spec_data_g430m.line_flux_spec['f4686']
    line_flux_err_f4686 = spec_data_g430m.line_flux_spec_err['f4686']

    print('target_id ', target_id)

    print('beta ', beta)
    print('beta_err ', beta_err)
    print('line_flux_f1640 ', line_flux_f1640)
    print('line_flux_err_f1640 ', line_flux_err_f1640)
    print('line_flux_f4686 ', line_flux_f4686)
    print('line_flux_err_f4686 ', line_flux_err_f4686)

    fit_dict = {
        'continuum_g140l': continuum_g140l.value,
        'wave_g140l': wave_g140l.value,
        'flux_g140l': flux_g140l.value,
        'flux_err_g140l': flux_err_g140l,
        'continuum_g430m': continuum_g430m.value,
        'wave_g430m': wave_g430m.value,
        'flux_g430m': flux_g430m.value,
        'flux_err_g430m': flux_err_g430m,
        'beta': beta,
        'beta_err': beta_err,
        'line_flux_f1640': line_flux_f1640,
        'line_flux_err_f1640': line_flux_err_f1640,
        'line_flux_f4686': line_flux_f4686,
        'line_flux_err_f4686': line_flux_err_f4686
    }

    np.save('data_output/fit_dict_%s.npy' % target_id, fit_dict)





