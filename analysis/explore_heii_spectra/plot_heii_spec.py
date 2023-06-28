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
emlines = pd.read_csv('/home/benutzer/Documents/projects/dust_analysis/uv spectral analysis/scripts/emlins.csv')
opt_cont_bad = np.array([(2850, 4000)]+[(tmp-5, tmp+5) for tmp in emlines.wavelength if (tmp>4000) and (tmp <5500)] + [(5500, 5700)])

# UV Continuum
# Windows to fit uv continuum
# Table 2 Calzetti et al 1994
# https://articles.adsabs.harvard.edu/pdf/1994ApJ...429..582C
calzetti_windows = np.array([(1265, 1285),
                             (1305, 1320),
                             (1345, 1370),
                             (1415, 1490),
                             (1565, 1585),
                             (1625, 1630),
                             (1658, 1673),
                             (1695, 1700)])
log_calzetti_windows = np.log10(calzetti_windows)
# optical fitting windows
span = [4000, 5500]
emlines = pd.read_csv('/home/benutzer/Documents/projects/dust_analysis/heII/scripts/final_reduction_scripts/optical_emlines.csv')
emline_windows = np.array((emlines.Line.values-10, emlines.Line.values+10)).T


def _log_spectra(flux, xlam, err):

    flux=np.where(flux>0, flux, np.nan) # negative numbers make errors in log-space

    ##
    log_flux=np.log10(flux)
    log_xlam=np.log10(xlam)
    log_err=err/(flux*np.log(10))

    return Spectrum1D(flux=log_flux*u_flux, spectral_axis=log_xlam*u_xlam, uncertainty=StdDevUncertainty(log_err, u_flux))


def _fit_continuum(flux, xlam, err, grating):

    if grating=='G140L':

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


u_flux = u.erg/u.s/u.cm/u.cm/u.AA
u_xlam = u.AA


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

for target in targets:
    print(target + 'A')
    spec = {b: _specData(target, b) for b in bands}
    all_spec[target + 'A'] = spec

    if glob.glob(f'/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/{target}/*B_sx1.fits') != []:
        print(target + 'B')
        spec = {b: _specData(target, b, tr='B') for b in bands}
        all_spec[target + 'B'] = spec

    if glob.glob(f'/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/{target}/*C_sx1.fits') != []:
        print(target + 'C')
        spec = {b: _specData(target, b, tr='C') for b in bands}
        all_spec[target + 'C'] = spec

    if glob.glob(f'/home/benutzer/Documents/projects/dust_analysis/heII/data/extracted_spec/{target}/*D_sx1.fits') != []:
        print(target + 'D')
        spec = {b: _specData(target, b, tr='D') for b in bands}
        all_spec[target + 'D'] = spec


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




for target_id in all_spec.keys():

    spec_data_g140l = all_spec[target_id]['G140L']
    spec_data_g430m = all_spec[target_id]['G430M']

    continuum_g140l = spec_data_g140l.continuum_flux
    spec_g140l = spec_data_g140l.spec1d
    wave_g140l = spec_g140l.spectral_axis
    flux_g140l = spec_g140l.flux
    flux_err_g140l = spec_data_g140l.err.array*u_flux
    fit_g140l = spec_data_g140l.fitted_flux[0]

    continuum_g430m = spec_data_g430m.continuum_flux
    spec_g430m = spec_data_g430m.spec1d
    wave_g430m = spec_g430m.spectral_axis
    flux_g430m = spec_g430m.flux
    flux_err_g430m = spec_data_g430m.err.array*u_flux
    fit_g430m = spec_data_g430m.fitted_flux[0]

    # fitting results
    beta = spec_data_g140l.continuum_model.slope.value
    beta_err = spec_data_g140l.continuum_model.stds['slope']
    line_flux_f1640 = spec_data_g140l.line_flux_spec['f1640']
    line_flux_err_f1640 = spec_data_g140l.line_flux_spec_err['f1640']
    line_flux_f4686 = spec_data_g430m.line_flux_spec['f4686']
    line_flux_err_f4686 = spec_data_g430m.line_flux_spec_err['f4686']

    print('spec_g140l ', spec_g140l)
    print('wave_g140l ', wave_g140l)
    print('flux_g140l ', flux_g140l)
    print('flux_err_g140l ', flux_err_g140l)

    print('beta ', beta)
    print('beta_err ', beta_err)
    print('line_flux_f1640 ', line_flux_f1640)
    print('line_flux_err_f1640 ', line_flux_err_f1640)
    print('line_flux_f4686 ', line_flux_f4686)
    print('line_flux_err_f4686 ', line_flux_err_f4686)



    figure = plt.figure(figsize=(25, 7))
    fontsize = 17
    ax_g140l = figure.add_axes([0.035, 0.08, 0.66, 0.88])
    ax_g430m = figure.add_axes([0.72, 0.08, 0.25, 0.88])
    ax_f1640 = figure.add_axes([0.49, 0.55, 0.18, 0.42])

    ax_g140l.step(wave_g140l, flux_g140l, where='mid', color='k', linewidth=2)

    # cont_mask = np.zeros(len(wave_g140l), dtype=bool)
    for window in calzetti_windows:
        mask_include_fit = (wave_g140l.value > window[0]) & (wave_g140l.value < window[1])
        # cont_mask += mask_include_fit
        ax_g140l.step(wave_g140l[mask_include_fit], flux_g140l[mask_include_fit], where='mid', color='tab:red', linewidth=2)


    ax_g140l.fill_between(wave_g140l.value, flux_g140l.value-flux_err_g140l.value, flux_g140l.value+flux_err_g140l.value,
                          color='gray', alpha=0.7)
    ax_g140l.plot([], [], color='white', label='G140L')
    ax_g140l.plot(wave_g140l, continuum_g140l, linestyle='--', linewidth=2, color='tab:blue', label=r'$\beta = %.2f \pm %.2f$' % (beta, beta_err))
    in_window = (wave_g140l > 1630 * u.AA) & (wave_g140l < 1652 * u.AA)
    ax_g140l.step(wave_g140l[in_window], flux_g140l[in_window], c='cyan', lw=2)


    # ax_g140l.axvspan(1630, 1652, color='k', hatch="////", )

    xlim_f1640 = (1640 - 50, 1640 + 50)

    mask_wave_f1640 = (wave_g140l.value > xlim_f1640[0]) & (wave_g140l.value < xlim_f1640[1])

    ax_f1640.step(wave_g140l[mask_wave_f1640], flux_g140l[mask_wave_f1640], where='mid', color='k', linewidth=2)
    ax_f1640.fill_between(wave_g140l.value[mask_wave_f1640], (flux_g140l.value-flux_err_g140l.value)[mask_wave_f1640],
                          (flux_g140l.value+flux_err_g140l.value)[mask_wave_f1640], color='gray', alpha=0.7)
    ax_f1640.plot(wave_g140l[mask_wave_f1640], continuum_g140l[mask_wave_f1640], linestyle='--', linewidth=2, color='tab:blue')
    ax_f1640.plot(wave_g140l[mask_wave_f1640], (continuum_g140l+fit_g140l)[mask_wave_f1640], linestyle='-', linewidth=2, color='tab:red')

    ax_f1640.step(wave_g140l[in_window], flux_g140l[in_window], c='cyan', lw=2)
    ax_f1640.set_xlim(xlim_f1640)

    mean_flux_f1640 = np.nanmean(flux_g140l.value[mask_wave_f1640])
    std_flux_f1640 = np.nanstd(flux_g140l.value[mask_wave_f1640])
    min_flux_f1640 = np.nanmin(flux_g140l.value[mask_wave_f1640])
    max_flux_f1640 = np.nanmax(flux_g140l.value[mask_wave_f1640])
    max_continuum_f1640 = np.nanmax(continuum_g140l.value[mask_wave_f1640])
    ax_f1640.set_ylim(min_flux_f1640 - std_flux_f1640, np.max([max_flux_f1640, max_continuum_f1640]) + std_flux_f1640)




    ax_g430m.step(wave_g430m, flux_g430m, where='mid', color='k', linewidth=2)
    ax_g430m.fill_between(wave_g430m.value, flux_g430m.value-flux_err_g430m.value, flux_g430m.value+flux_err_g430m.value,
                          color='gray', alpha=0.7)
    ax_g430m.plot(wave_g430m, continuum_g430m, linestyle='--', linewidth=2, color='tab:blue')
    ax_g430m.plot(wave_g430m, continuum_g430m + fit_g430m, linestyle='-', linewidth=2, color='tab:red')
    in_window = (wave_g430m > 4670 * u.AA) & (wave_g430m < 4705 * u.AA)
    ax_g430m.step(wave_g430m[in_window], flux_g430m[in_window], c='cyan', lw=2)


    mean_flux_g140l = np.nanmean(flux_g140l.value)
    std_flux_g140l = np.nanstd(flux_g140l.value)
    min_flux_g140l = np.nanmin(flux_g140l.value)
    max_flux_g140l = np.nanmax(flux_g140l.value)
    max_continuum_g140l = np.nanmax(continuum_g140l.value)
    ax_g140l.set_ylim(min_flux_g140l - std_flux_g140l, np.max([max_flux_g140l, max_continuum_g140l]) + 5*std_flux_g140l)


    xlim_g430m = (4686 - 100, 4686 + 100)
    mask_wave_g430m = (wave_g430m.value > xlim_g430m[0]) & (wave_g430m.value < xlim_g430m[1])

    mean_flux_g430m = np.nanmean(flux_g430m.value[mask_wave_g430m])
    std_flux_g430m = np.nanstd(flux_g430m.value[mask_wave_g430m])
    min_flux_g430m = np.nanmin(flux_g430m.value[mask_wave_g430m])
    max_flux_g430m = np.nanmax(flux_g430m.value[mask_wave_g430m])
    max_continuum_g430m = np.nanmax(continuum_g430m.value[mask_wave_g430m])
    ax_g430m.set_ylim(min_flux_g430m - 1*std_flux_g430m, np.max([max_flux_g430m, max_continuum_g430m]) + 3*std_flux_g430m)

    ax_g140l.set_xlim(1150, 1700)
    ax_g430m.set_xlim(xlim_g430m)




    ax_g140l.fill_between([1630, 1652], ax_g140l.get_ylim()[1], ax_g140l.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")
    ax_f1640.fill_between([1630, 1652], ax_f1640.get_ylim()[1], ax_f1640.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")
    ax_g430m.fill_between([4670, 4705], ax_g430m.get_ylim()[1], ax_g430m.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")

    flux_str_f1640 = r'HeII(1640) = %.2f $^{+%.2f}_{-%.2f} \times 10^{-15}$ erg/s/cm$^{2}$ ' % (line_flux_f1640, line_flux_err_f1640[0], line_flux_err_f1640[1])
    ax_f1640.text(ax_f1640.get_xlim()[0] + (ax_f1640.get_xlim()[1] - ax_f1640.get_xlim()[0]) * 0.03,
                  ax_f1640.get_ylim()[0] + (ax_f1640.get_ylim()[1] - ax_f1640.get_ylim()[0]) * 0.9,
                  flux_str_f1640, fontsize=fontsize-4)

    ax_g140l.legend(frameon=False, loc=2, fontsize=fontsize)

    flux_str_f4686 = r'HeII(4686) = %.2f $^{+%.2f}_{-%.2f} \times 10^{-15}$ erg/s/cm$^{2}$ ' % \
                    (line_flux_f4686, line_flux_err_f4686[0], line_flux_err_f4686[1])

    ax_g430m.text(ax_g430m.get_xlim()[0] + (ax_g430m.get_xlim()[1] - ax_g430m.get_xlim()[0]) * 0.03,
                  ax_g430m.get_ylim()[0] + (ax_g430m.get_ylim()[1] - ax_g430m.get_ylim()[0]) * 0.93,
                  'G430M', fontsize=fontsize)
    ax_g430m.text(ax_g430m.get_xlim()[0] + (ax_g430m.get_xlim()[1] - ax_g430m.get_xlim()[0]) * 0.03,
                  ax_g430m.get_ylim()[0] + (ax_g430m.get_ylim()[1] - ax_g430m.get_ylim()[0]) * 0.865,
                  flux_str_f4686, fontsize=fontsize)


    ax_g140l.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_f1640.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_g430m.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
    ax_g140l.yaxis.offsetText.set_fontsize(fontsize - 3)
    ax_f1640.yaxis.offsetText.set_fontsize(fontsize - 3)
    ax_g430m.yaxis.offsetText.set_fontsize(fontsize - 3)



    ax_g140l.set_xlabel('Rest Wavelength (Å)', labelpad=-3, fontsize=fontsize)
    ax_g430m.set_xlabel('Rest Wavelength (Å)', labelpad=-3, fontsize=fontsize)
    ax_g140l.set_ylabel('Flux (erg/s/cm/cm/Å)', labelpad=-15, fontsize=fontsize)





    plt.savefig('plot_output/spec_result_%s.png' % target_id)
    plt.cla()


exit()




print([k for k in keys if k[:-1] == targ])

for t in targets:
    print('t ', t)
    clusters = [k for k in keys if k[:-1] == t]

    cnt = 0
    for ii,cid in enumerate(clusters):
        print('ii,cid ', ii,cid)
        spec_data=all_spec[cid]

        fig = plt.figure(figsize=(8,4), dpi=100, facecolor='w') #(22,30))#(values5))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        fig.patch.set_facecolor('w')

        for ii, line_dat in plot_dict.items():
            print('ii, line_dat ', ii, line_dat)

            b, line, window, xlim = line_dat.values()
            print('b, line, window, xlim ', b, line, window, xlim)

            if b!='G430L':
                dat=all_spec[cid][b] #t+'A'][b]

                continuum=dat.continuum_flux

                spec=dat.spec1d
                X=spec.spectral_axis
                Y=spec.flux
                E=dat.err.array*u_flux


                ## plot
                ax=plt.subplot(1,2,ii+1)
                ax.step(X, Y, label='Spectra', c='k', lw=1)
                ax.plot(X, continuum, label='Continuum', c='gray', lw=1.15, ls='-')
                plt.fill_between(X, Y-E, Y+E, step="pre",color=(0.7,0.7,0.7,0.5))

                 ### get ylimit
                in_frame= np.where((X.value>=xlim[0]) & (X.value<=xlim[1]), True, False)
                flux_in_frame=Y[in_frame].value

                # get rms & std
                rms, std =np.sqrt(np.mean(flux_in_frame**2)), np.std(flux_in_frame)
                ylim=[rms-8*std, rms+8*std]

                ax.set_ylim(*ylim)
                ax.set_xlim(*xlim)

                fit_models=dat.line_model
                for m_idx, m in enumerate(fit_models):

                    fit=m(X)
                    m_window=window[m_idx]

                    #### highlight emission lines
                    in_window=[ii for ii,x in enumerate(X) if (x>=m_window[0])&(x<=m_window[1])]
                    non_zero=[ii for ii,x in enumerate(X.value) if (x>=m_window[0].value-35)&(x<=m_window[1].value+35)]  #[ii for ii,ft in enumerate(fit) if (ft-continuum[ii]>0)]

                    ax.step(X[in_window], Y[in_window], c='cyan', lw=1) #'navy', lw=1)

                    if m_idx!=1:
    #                     ax.plot(X[non_zero], fit[non_zero]+continuum[non_zero], c='cyan', lw=1.5, ls='-', label=f'Line Fit')
                        ax.fill_between([m_window[0].value, m_window[1].value], ax.get_ylim()[1], ax.get_ylim()[0], color="none", edgecolor=(0.9,0.9,0.9), hatch="////")


                ### labels
                if ii==0:
                    ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                    ax.set_ylabel('Flux (erg/s/cm/cm/Å)', fontsize=12)
                elif ii==1:
                    ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                    ax.set_ylabel('')
                elif ii==2:
                    ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                    ax.set_ylabel('', fontsize=12)
                else:
                    ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                    ax.set_ylabel('', fontsize=12)

                if ii==3:
                    ax.legend(loc='lower right', fontsize='small')       # add legend

                ### plot labels
                ymin,ymax=ax.get_ylim()
                xmin,xmax=ax.get_xlim()

                 # add text
                tmp_flux=np.round(dat.line_flux_spec[line],2)
                tmp_err_high, tmp_err_low=[np.round(tmp, 2) for tmp in dat.line_flux_spec_err[line]]
                if line=='f1640': ln='F(1640)'
                elif line=='f4686': ln='F(4686)'
                elif line=='hb': ln='F(Hβ)'
                elif line=='hg': ln='F(Hγ)'

                txt = f"${ln} = {tmp_flux}" + " ^{+" + str(tmp_err_high) + "}_{-" + str(tmp_err_low) + "}\\times 10^{-15}\,{\\rm erg/s/cm^2}$"
                textplt1 = ax.text(xmin + (xmax-xmin)*0.05, ymin + (ymax-ymin)*0.93,
                                    txt,
                                    ha='left', va='center',fontsize=10)

                ax.text(xmin + (xmax-xmin)*0.03, ymin+(ymax-ymin)*0.02, cid, ha='left', va='bottom', fontsize=16, color='k')
                ax.text(xmin + (xmax-xmin)*0.03, ymin+(ymax-ymin)*0.10, ln, ha='left', va='bottom', fontsize=12)

                if b!='G140L':
                    for iii,row in emlines.iterrows():
                        ax.axvline(row['Line'], c='r', alpha=0.3, lw=0.5)
                        ax.set_xlim(*xlim)
                        ax.set_ylim(*ylim)

            fig.tight_layout(w_pad=0)
            fig.savefig(f'plot_output/{cid}_row_fit_all.png')


# PDF = PdfPages('zoom_all_spec.pdf')
PDF = PdfPages('zoom2_spec.pdf')


# t = targets[1]
for t in targets:
    clusters = [k for k in keys if k[:-1]==t]

    cnt=0
    for ii,cid in enumerate(clusters):

        spec_data=all_spec[cid]

        fig=plt.figure(figsize=(16,4), dpi=100, facecolor='w') #(22,30))#(values5))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
        fig.patch.set_facecolor('w')

        for ii, line_dat in plot_dict.items():

            b, line, window, xlim = line_dat.values()
            dat=all_spec[cid][b] #t+'A'][b]

            continuum=dat.continuum_flux

            spec=dat.spec1d
            X=spec.spectral_axis
            Y=spec.flux
            E=dat.err.array*u_flux


            ## plot
            ax=plt.subplot(1,4,ii+1)
            ax.step(X, Y, label='Spectra', c='k', lw=1)
            ax.plot(X, continuum, label='Continuum', c='gray', lw=1.15, ls='-')
            plt.fill_between(X, Y-E, Y+E, step="pre",color=(0.7,0.7,0.7,0.5))

             ### get ylimit
            in_frame= np.where((X.value>=xlim[0]) & (X.value<=xlim[1]), True, False)
            flux_in_frame=Y[in_frame].value

            # get rms & std
            rms, std =np.sqrt(np.mean(flux_in_frame**2)), np.std(flux_in_frame)
            ylim=[rms-8*std, rms+8*std]

            ax.set_ylim(*ylim)
#             ax.set_xlim(*xlim)

            fit_models=dat.line_model
            for m_idx, m in enumerate(fit_models):

                fit=m(X)
                m_window=window[m_idx]

                #### highlight emission lines
                in_window=[ii for ii,x in enumerate(X) if (x>=m_window[0])&(x<=m_window[1])]
                non_zero=[ii for ii,x in enumerate(X.value) if (x>=m_window[0].value-35)&(x<=m_window[1].value+35)]  #[ii for ii,ft in enumerate(fit) if (ft-continuum[ii]>0)]

                ax.step(X[in_window], Y[in_window], c='cyan', lw=1) #'navy', lw=1)

                if m_idx!=1:
#                     ax.plot(X[non_zero], fit[non_zero]+continuum[non_zero], c='cyan', lw=1.5, ls='-', label=f'Line Fit')
                    ax.fill_between([m_window[0].value, m_window[1].value], ax.get_ylim()[1], ax.get_ylim()[0], color="none", edgecolor=(0.9,0.9,0.9), hatch="////")


            ### labels
            if ii==0:
                ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                ax.set_ylabel('Flux (erg/s/cm/cm/Å)', fontsize=12)
            elif ii==1:
                ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                ax.set_ylabel('')
            elif ii==2:
                ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                ax.set_ylabel('', fontsize=12)
            else:
                ax.set_xlabel('Rest Wavelength (Å)', fontsize=12)
                ax.set_ylabel('', fontsize=12)

            if ii==3:
                ax.legend(loc='lower right', fontsize='small')       # add legend

            ### plot labels
            ymin,ymax=ax.get_ylim()
            xmin,xmax=ax.get_xlim()

             # add text
            tmp_flux=np.round(dat.line_flux_spec[line],2)
            tmp_err_high, tmp_err_low=[np.round(tmp, 2) for tmp in dat.line_flux_spec_err[line]]
            if line=='f1640': ln='F(1640)'
            elif line=='f4686': ln='F(4686)'
            elif line=='hb': ln='F(Hβ)'
            elif line=='hg': ln='F(Hγ)'

#             txt = f"${ln} = {tmp_flux}" + " ^{+" + str(tmp_err_high) + "}_{-" + str(tmp_err_low) + "}\\times 10^{-15}\,{\\rm erg/s/cm^2}$"
#             textplt1 = ax.text(xmin + (xmax-xmin)*0.05, ymin + (ymax-ymin)*0.93,
#                                 txt,
#                                 ha='left', va='center',fontsize=8)
            if b=='G430L':
                xmin,xmax= [4300, 5000]
            ax.text(xmin + (xmax-xmin)*0.03, ymin+(ymax-ymin)*0.02, cid, ha='left', va='bottom', fontsize=16, color='k')
            ax.text(xmin + (xmax-xmin)*0.03, ymin+(ymax-ymin)*0.10, ln, ha='left', va='bottom', fontsize=12)

            if b!='G140L':
                ax.vlines(emlines.Line.values, ymin, ymax, color='r', linestyles='--', alpha=0.5, lw=0.5)
# #                 for iii,row in emlines.iterrows():
# #                     ax.axvline(row['Line'], c='r', alpha=0.3, lw=0.5, ls='--')
# #                     ax.set_ylim(*ylim)

                ax.set_xlim(xmin,xmax)


        fig.tight_layout(w_pad=0)

        PDF.savefig(fig)
        plt.close()
    fig.savefig(f'plot_output/{cid}_row_fit_zoom.png')
PDF.close()


