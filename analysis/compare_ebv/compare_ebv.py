

import os , glob
import numpy as np
import pandas as pd

import dust_tools.extinction_tools

np.seterr(all='ignore')  # hides irrelevant warnings about divide-by-zero, etc

from pathlib import Path    # handle paths to files

import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
from astropy.modeling import fitting
from astropy.nddata import StdDevUncertainty
from astropy.nddata import NDData
from astropy.wcs import WCS

from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.fitting import fit_lines
from specutils.fitting import fit_continuum
from specutils.manipulation import extract_region

from dust_extinction.parameter_averages import CCM89

# %matplotlib inline
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects

from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)


def k_lambda(wavelength, law='R15'):
    # Only valid for wavelengths < 0.6μm, wavelength input in microns
    # R15: Reddy et al. 2015, C00: Calzetti et al. 2000
    params={'R15':{'Rv':2.505, 'coeff':np.array([-5.729, 4.004, 0.525, 0.029])},
            'C00':{'Rv':4.050, 'coeff':2.659*np.array([-2.156, 1.509, 0.198, 0.011])}}

    Rv=params[law]['Rv']
    a,b,c,d=params[law]['coeff']

    return  a + b/wavelength - c/wavelength**2 + d/wavelength**3 + Rv

# Ε(Β-V)
def E_BV_ratio(observed_ratio, line_ratio, err=0, reddening_law='R15'):
    # E(B-V) = 2.5/[k(λ2)-k(λ1)] * log(R_obs/R0) = 2.5/[k(λ2)-k(λ1)] * [log(R_obs) - log(R0)]
    #        = A * log(R_obs) - B
    if line_ratio=='heII':
        l1, l2 = 1640e-4, 4686e-4
        log_R0 = 0.89

    elif line_ratio=='balmer':
        l1, l2 = [4341e-4, 4861e-4]
        log_R0=np.log10(0.47)
    A = 2.5/(k_lambda(l2, reddening_law) - k_lambda(l1, reddening_law)) #(absorption_coeff(l2, reddening_law) - absorption_coeff(l1, reddening_law))
    B = A*log_R0
    EBV = A*np.log10(observed_ratio) - B

    ## ERROR
    d_ratio=A/(observed_ratio*np.log(10))
    EBV_err=np.sqrt(d_ratio**2 * err**2)

    return EBV, EBV_err

## UV slope:
def E_BV_uv(beta, err):
    EBV=(beta+2.44)/4.54

    d_beta=1/4.54
    EBV_err=np.sqrt(err**2 * d_beta**2)

    return EBV, EBV_err


calzetti_windows = np.array([(1265, 1285),
                            (1305, 1320),
                            (1345, 1370),
                            (1415, 1490),
                            (1565, 1585),
                            (1625, 1630),
                            (1658, 1673),
                            (1695, 1700)])

plotting_params = {
    'HE_2_10A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '+', 'color': 'tab:blue'},
    'HE_2_10B': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '+', 'color': 'tab:blue'},
    'HE_2_10C': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '+', 'color': 'tab:blue'},
    'HE_2_10D': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '+', 'color': 'tab:blue'},
    'MRK_33A': {
        'heii_1640_det': False, 'heii_4686_det': False, 'marker': 'D', 'color': 'tab:orange'},
    'MRK_33B': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': 'D', 'color': 'tab:orange'},
    'NGC_3049A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '*', 'color': 'tab:green'},
    'NGC_3049B': {
        'heii_1640_det': False, 'heii_4686_det': False, 'marker': '*', 'color': 'tab:green'},
    'NGC_3125A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': 'o', 'color': 'tab:red'},
    'NGC_4214A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': 'p', 'color': 'tab:purple'},
    'NGC_4670A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': 's', 'color': 'tab:brown'},
    'NGC_4670B': {
        'heii_1640_det': False, 'heii_4686_det': False, 'marker': 's', 'color': 'tab:brown'},
    'NGC_4670C': {
        'heii_1640_det': False, 'heii_4686_det': False, 'marker': 's', 'color': 'tab:brown'},
    'TOL_1924_416A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': 'x', 'color': 'tab:pink'},
    'TOL_1924_416B': {
        'heii_1640_det': False, 'heii_4686_det': False, 'marker': 'x', 'color': 'tab:pink'},
    'TOL_89A': {
        'heii_1640_det': True, 'heii_4686_det': True, 'marker': '>', 'color': 'tab:gray'}
}


# add Av to df
Av_dict = {'HE_2_10': 0.306,
           'NGC_3049': 0.105,
           'NGC_3125': 0.21,
           'MRK_33': 0.033,
           'NGC_4214': 0.06,
           'NGC_4670': 0.041,
           'TOL_89': 0.181,
           'TOL_1924_416': 0.236}

def mw_de_redden_flux(av, obs_flux, obs_xlam):
    # Correct for interstellar reddening from milky way using Mathis 1990 law
    # xlam -- Angstrom (observed wavelength NOT restframe)
    # flux -- erg/s/cm/cm/Angstrom
    ext = CCM89(Rv=3.1)
    dust = ext.extinguish(obs_xlam, Av=av)

    return obs_flux/dust

def get_ebv_stis_values(target_name):
    fit_dict = np.load('../heii_spectra/data_output/fit_dict_%s.npy' % target_name, allow_pickle=True).item()
    # get data
    beta = fit_dict['beta']
    beta_err = fit_dict['beta_err']
    line_flux_f1640 = fit_dict['line_flux_f1640']
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686']
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']
    he_line_ratio = line_flux_f1640 / line_flux_f4686
    mean_err_f1640 = np.mean(line_flux_err_f1640)
    mean_err_f4686 = np.mean(line_flux_err_f4686)
    he_line_ratio_err = np.sqrt((mean_err_f1640 / line_flux_f1640) ** 2 +
                                ((mean_err_f4686 * line_flux_f1640) / (line_flux_f4686 ** 2)) ** 2)
    ebv_uv, ebv_uv_err = E_BV_uv(beta=beta, err=beta_err)
    ebv_heii, ebv_heii_err = E_BV_ratio(observed_ratio=he_line_ratio, line_ratio='heII', err=he_line_ratio_err, reddening_law='R15')
    return ebv_uv, ebv_uv_err, ebv_heii, ebv_heii_err


def get_ebv_balmer_values(file_name, av):
    fit_dict = np.load(file_name, allow_pickle=True).item()

    if 'h_alpha_flux' in fit_dict.keys():
        h_alpha_flux = fit_dict['h_alpha_flux']
        h_beta_flux = fit_dict['h_beta_flux']
        h_alpha_flux_err = fit_dict['h_alpha_flux_err']
        h_beta_flux_err = fit_dict['h_beta_flux_err']

        h_alpha_flux_de_red = mw_de_redden_flux(av=av, obs_flux=h_alpha_flux,
                                                obs_xlam=6565*u.AA)
        h_beta_flux_de_red = mw_de_redden_flux(av=av, obs_flux=h_beta_flux,
                                                obs_xlam=6565*u.AA)

        ebv = dust_tools.extinction_tools.ExtinctionTools.get_balmer_extinct_alpha_beta(flux_h_alpha_6565=h_alpha_flux_de_red,
                                                                                  flux_h_beta_4863=h_beta_flux_de_red)
        ebv_err = dust_tools.extinction_tools.ExtinctionTools.get_balmer_extinct_alpha_beta_err(flux_h_alpha_6565=h_alpha_flux_de_red,
                                                                                  flux_h_beta_4863=h_beta_flux_de_red,
                                                                                                flux_h_alpha_6565_err=h_alpha_flux_err,
                                                                                                flux_h_beta_4863_err=h_beta_flux_err)
    else:
        h_gamma_flux = fit_dict['h_gamma_flux']
        h_beta_flux = fit_dict['h_beta_flux']
        h_gamma_flux_err = fit_dict['h_gamma_flux_err']
        h_beta_flux_err = fit_dict['h_beta_flux_err']

        h_gamma_flux_de_red = mw_de_redden_flux(av=av, obs_flux=h_gamma_flux,
                                                obs_xlam=4341*u.AA)
        h_beta_flux_de_red = mw_de_redden_flux(av=av, obs_flux=h_beta_flux,
                                                obs_xlam=6565*u.AA)

        ebv = dust_tools.extinction_tools.ExtinctionTools.get_balmer_extinct_beta_gamma(flux_h_gamma_4342=h_gamma_flux_de_red,
                                                                                  flux_h_beta_4863=h_beta_flux_de_red)
        ebv_err = dust_tools.extinction_tools.ExtinctionTools.get_balmer_extinct_beta_gamma_err(flux_h_gamma_4342=h_gamma_flux_de_red,
                                                                                  flux_h_beta_4863=h_beta_flux_de_red,
                                                                                                flux_h_gamma_4342_err=h_gamma_flux_err,
                                                                                                flux_h_beta_4863_err=h_beta_flux_err)

    return ebv, ebv_err


def get_balmer_ebv(target, ext='A'):
    ebv_balmer, ebv_balmer_err = get_ebv_balmer_values(
        file_name='../balmer/%s/data_output/balmer_dict_%s.npy' % (target.lower(), target.upper() + ext),
        av=Av_dict[target.upper()])
    return ebv_balmer, ebv_balmer_err


#HE 2-10
ebv_uv_he210a, ebv_uv_err_he210a, ebv_heii_he210a, ebv_heii_err_he210a = get_ebv_stis_values(target_name='HE_2_10A')
ebv_uv_he210b, ebv_uv_err_he210b, ebv_heii_he210b, ebv_heii_err_he210b = get_ebv_stis_values(target_name='HE_2_10B')
ebv_uv_he210c, ebv_uv_err_he210c, ebv_heii_he210c, ebv_heii_err_he210c = get_ebv_stis_values(target_name='HE_2_10C')
ebv_uv_he210d, ebv_uv_err_he210d, ebv_heii_he210d, ebv_heii_err_he210d = get_ebv_stis_values(target_name='HE_2_10D')
ebv_balmer_he210a, ebv_balmer_err_he210a = get_balmer_ebv(target='HE_2_10', ext='A')
ebv_balmer_he210b, ebv_balmer_err_he210b = get_balmer_ebv(target='HE_2_10', ext='B')
ebv_balmer_he210c, ebv_balmer_err_he210c = get_balmer_ebv(target='HE_2_10', ext='C')
ebv_balmer_he210d, ebv_balmer_err_he210d = get_balmer_ebv(target='HE_2_10', ext='D')

ebv_uv_mrk33b, ebv_uv_err_mrk33b, ebv_heii_mrk33b, ebv_heii_err_mrk33b = get_ebv_stis_values(target_name='MRK_33B')
ebv_balmer_mrk33, ebv_balmer_err_mrk33 = get_balmer_ebv(target='MRK_33', ext='A')

ebv_uv_ngc3049a, ebv_uv_err_ngc3049a, ebv_heii_ngc3049a, ebv_heii_err_ngc3049a = get_ebv_stis_values(target_name='NGC_3049A')
ebv_balmer_ngc3049, ebv_balmer_err_ngc3049 = get_balmer_ebv(target='NGC_3049', ext='')

ebv_uv_ngc3125a, ebv_uv_err_ngc3125a, ebv_heii_ngc3125a, ebv_heii_err_ngc3125a = get_ebv_stis_values(target_name='NGC_3125A')
ebv_balmer_ngc3125, ebv_balmer_err_ngc3125 = get_balmer_ebv(target='NGC_3125', ext='A')

ebv_uv_ngc4214a, ebv_uv_err_ngc4214a, ebv_heii_ngc4214a, ebv_heii_err_ngc4214a = get_ebv_stis_values(target_name='NGC_4214A')
ebv_balmer_ngc4214, ebv_balmer_err_ngc4214 = get_balmer_ebv(target='NGC_4214', ext='')

ebv_uv_ngc4670a, ebv_uv_err_ngc4670a, ebv_heii_ngc4670a, ebv_heii_err_ngc4670a = get_ebv_stis_values(target_name='NGC_4670A')
ebv_balmer_ngc4670, ebv_balmer_err_ngc4670 = get_balmer_ebv(target='NGC_4670', ext='')

ebv_uv_tol1924a, ebv_uv_err_tol1924a, ebv_heii_tol1924a, ebv_heii_err_tol1924a = get_ebv_stis_values(target_name='TOL_1924_416A')
ebv_balmer_tol1924, ebv_balmer_err_tol1924 = get_balmer_ebv(target='TOL_1924_416', ext='')

ebv_uv_tol89a, ebv_uv_err_tol89a, ebv_heii_tol89a, ebv_heii_err_tol89a = get_ebv_stis_values(target_name='TOL_89A')
ebv_balmer_tol89, ebv_balmer_err_tol89 = get_balmer_ebv(target='TOL_89', ext='')


figure = plt.figure(figsize=(25, 20))
fontsize = 28
ax_balmer_heii = figure.add_axes([0.065, 0.53, 0.465, 0.465])
ax_balmer_uv = figure.add_axes([0.53, 0.53, 0.465, 0.465])
ax_heii_uv = figure.add_axes([0.53, 0.065, 0.465, 0.465])
ax_blank = figure.add_axes([0.065, 0.065, 0.465, 0.465])
ax_blank.axis('off')

ax_balmer_heii.scatter(ebv_heii_he210a, ebv_balmer_he210a, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_heii.scatter(ebv_heii_he210b, ebv_balmer_he210b, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_heii.scatter(ebv_heii_he210c, ebv_balmer_he210c, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_heii.scatter(ebv_heii_he210d, ebv_balmer_he210d, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_he210a, ebv_balmer_he210a, xerr=ebv_heii_err_he210a, yerr=ebv_balmer_err_he210a,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_heii.errorbar(ebv_heii_he210b, ebv_balmer_he210b, xerr=ebv_heii_err_he210b, yerr=ebv_balmer_err_he210b,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_heii.errorbar(ebv_heii_he210c, ebv_balmer_he210c, xerr=ebv_heii_err_he210c, yerr=ebv_balmer_err_he210c,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_heii.errorbar(ebv_heii_he210d, ebv_balmer_he210d, xerr=ebv_heii_err_he210d, yerr=ebv_balmer_err_he210d,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])


ax_balmer_uv.scatter(ebv_uv_he210a, ebv_balmer_he210a, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_uv.scatter(ebv_uv_he210b, ebv_balmer_he210b, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_uv.scatter(ebv_uv_he210c, ebv_balmer_he210c, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_uv.scatter(ebv_uv_he210d, ebv_balmer_he210d, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_he210a, ebv_balmer_he210a, xerr=ebv_uv_err_he210a, yerr=ebv_balmer_err_he210a,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_uv.errorbar(ebv_uv_he210b, ebv_balmer_he210b, xerr=ebv_uv_err_he210b, yerr=ebv_balmer_err_he210b,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_uv.errorbar(ebv_uv_he210c, ebv_balmer_he210c, xerr=ebv_uv_err_he210c, yerr=ebv_balmer_err_he210c,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_balmer_uv.errorbar(ebv_uv_he210d, ebv_balmer_he210d, xerr=ebv_uv_err_he210d, yerr=ebv_balmer_err_he210d,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])

ax_heii_uv.scatter(ebv_uv_he210a, ebv_heii_he210a, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_heii_uv.scatter(ebv_uv_he210b, ebv_heii_he210b, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_heii_uv.scatter(ebv_uv_he210c, ebv_heii_he210c, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_heii_uv.scatter(ebv_uv_he210d, ebv_heii_he210d, s=150, color=plotting_params['HE_2_10A']['color'],
                      marker=plotting_params['HE_2_10A']['marker'])
ax_heii_uv.errorbar(ebv_uv_he210a, ebv_heii_he210a, xerr=ebv_uv_err_he210a, yerr=ebv_heii_err_he210a,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_heii_uv.errorbar(ebv_uv_he210b, ebv_heii_he210b, xerr=ebv_uv_err_he210b, yerr=ebv_heii_err_he210b,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_heii_uv.errorbar(ebv_uv_he210c, ebv_heii_he210c, xerr=ebv_uv_err_he210c, yerr=ebv_heii_err_he210c,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])
ax_heii_uv.errorbar(ebv_uv_he210d, ebv_heii_he210d, xerr=ebv_uv_err_he210d, yerr=ebv_heii_err_he210d,
                       fmt='.', color=plotting_params['HE_2_10A']['color'])




ax_balmer_heii.scatter(ebv_heii_mrk33b, ebv_balmer_mrk33, s=150, color=plotting_params['MRK_33A']['color'],
                      marker=plotting_params['MRK_33A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_mrk33b, ebv_balmer_mrk33, xerr=ebv_heii_err_mrk33b, yerr=ebv_balmer_err_mrk33,
                       fmt='.', color=plotting_params['MRK_33A']['color'])
ax_balmer_uv.scatter(ebv_uv_mrk33b, ebv_balmer_mrk33, s=150, color=plotting_params['MRK_33A']['color'],
                      marker=plotting_params['MRK_33A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_mrk33b, ebv_balmer_mrk33, xerr=ebv_uv_err_mrk33b, yerr=ebv_balmer_err_mrk33,
                       fmt='.', color=plotting_params['MRK_33A']['color'])
ax_heii_uv.scatter(ebv_uv_mrk33b, ebv_heii_mrk33b, s=150, color=plotting_params['MRK_33A']['color'],
                      marker=plotting_params['MRK_33A']['marker'])
ax_heii_uv.errorbar(ebv_uv_mrk33b, ebv_heii_mrk33b, xerr=ebv_uv_err_mrk33b, yerr=ebv_heii_err_mrk33b,
                       fmt='.', color=plotting_params['MRK_33A']['color'])


ax_balmer_heii.scatter(ebv_heii_ngc3049a, ebv_balmer_ngc3049, s=150, color=plotting_params['NGC_3049A']['color'],
                      marker=plotting_params['NGC_3049A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_ngc3049a, ebv_balmer_ngc3049, xerr=ebv_heii_err_ngc3049a, yerr=ebv_balmer_err_ngc3049,
                       fmt='.', color=plotting_params['NGC_3049A']['color'])
ax_balmer_uv.scatter(ebv_uv_ngc3049a, ebv_balmer_ngc3049, s=150, color=plotting_params['NGC_3049A']['color'],
                      marker=plotting_params['NGC_3049A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_ngc3049a, ebv_balmer_ngc3049, xerr=ebv_uv_err_ngc3049a, yerr=ebv_balmer_err_ngc3049,
                       fmt='.', color=plotting_params['NGC_3049A']['color'])
ax_heii_uv.scatter(ebv_uv_ngc3049a, ebv_heii_ngc3049a, s=150, color=plotting_params['NGC_3049A']['color'],
                      marker=plotting_params['NGC_3049A']['marker'])
ax_heii_uv.errorbar(ebv_uv_ngc3049a, ebv_heii_ngc3049a, xerr=ebv_uv_err_ngc3049a, yerr=ebv_heii_err_ngc3049a,
                       fmt='.', color=plotting_params['NGC_3049A']['color'])



ax_balmer_heii.scatter(ebv_heii_ngc3125a, ebv_balmer_ngc3125, s=150, color=plotting_params['NGC_3125A']['color'],
                      marker=plotting_params['NGC_3125A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_ngc3125a, ebv_balmer_ngc3125, xerr=ebv_heii_err_ngc3125a, yerr=ebv_balmer_err_ngc3125,
                       fmt='.', color=plotting_params['NGC_3125A']['color'])
ax_balmer_uv.scatter(ebv_uv_ngc3125a, ebv_balmer_ngc3125, s=150, color=plotting_params['NGC_3125A']['color'],
                      marker=plotting_params['NGC_3125A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_ngc3125a, ebv_balmer_ngc3125, xerr=ebv_uv_err_ngc3125a, yerr=ebv_balmer_err_ngc3125,
                       fmt='.', color=plotting_params['NGC_3125A']['color'])
ax_heii_uv.scatter(ebv_uv_ngc3125a, ebv_heii_ngc3125a, s=150, color=plotting_params['NGC_3125A']['color'],
                      marker=plotting_params['NGC_3125A']['marker'])
ax_heii_uv.errorbar(ebv_uv_ngc3125a, ebv_heii_ngc3125a, xerr=ebv_uv_err_ngc3125a, yerr=ebv_heii_err_ngc3125a,
                       fmt='.', color=plotting_params['NGC_3125A']['color'])


ax_balmer_heii.scatter(ebv_heii_ngc4214a, ebv_balmer_ngc4214, s=150, color=plotting_params['NGC_4214A']['color'],
                      marker=plotting_params['NGC_4214A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_ngc4214a, ebv_balmer_ngc4214, xerr=ebv_heii_err_ngc4214a, yerr=ebv_balmer_err_ngc4214,
                       fmt='.', color=plotting_params['NGC_4214A']['color'])
ax_balmer_uv.scatter(ebv_uv_ngc4214a, ebv_balmer_ngc4214, s=150, color=plotting_params['NGC_4214A']['color'],
                      marker=plotting_params['NGC_4214A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_ngc4214a, ebv_balmer_ngc4214, xerr=ebv_uv_err_ngc4214a, yerr=ebv_balmer_err_ngc4214,
                       fmt='.', color=plotting_params['NGC_4214A']['color'])
ax_heii_uv.scatter(ebv_uv_ngc4214a, ebv_heii_ngc4214a, s=150, color=plotting_params['NGC_4214A']['color'],
                      marker=plotting_params['NGC_4214A']['marker'])
ax_heii_uv.errorbar(ebv_uv_ngc4214a, ebv_heii_ngc4214a, xerr=ebv_uv_err_ngc4214a, yerr=ebv_heii_err_ngc4214a,
                       fmt='.', color=plotting_params['NGC_4214A']['color'])


ax_balmer_heii.scatter(ebv_heii_ngc4670a, ebv_balmer_ngc4670, s=150, color=plotting_params['NGC_4670A']['color'],
                      marker=plotting_params['NGC_4670A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_ngc4670a, ebv_balmer_ngc4670, xerr=ebv_heii_err_ngc4670a, yerr=ebv_balmer_err_ngc4670,
                       fmt='.', color=plotting_params['NGC_4670A']['color'])
ax_balmer_uv.scatter(ebv_uv_ngc4670a, ebv_balmer_ngc4670, s=150, color=plotting_params['NGC_4670A']['color'],
                      marker=plotting_params['NGC_4670A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_ngc4670a, ebv_balmer_ngc4670, xerr=ebv_uv_err_ngc4670a, yerr=ebv_balmer_err_ngc4670,
                       fmt='.', color=plotting_params['NGC_4670A']['color'])
ax_heii_uv.scatter(ebv_uv_ngc4670a, ebv_heii_ngc4670a, s=150, color=plotting_params['NGC_4670A']['color'],
                      marker=plotting_params['NGC_4670A']['marker'])
ax_heii_uv.errorbar(ebv_uv_ngc4670a, ebv_heii_ngc4670a, xerr=ebv_uv_err_ngc4670a, yerr=ebv_heii_err_ngc4670a,
                       fmt='.', color=plotting_params['NGC_4670A']['color'])


ax_balmer_heii.scatter(ebv_heii_tol1924a, ebv_balmer_tol1924, s=150, color=plotting_params['TOL_1924_416A']['color'],
                      marker=plotting_params['TOL_1924_416A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_tol1924a, ebv_balmer_tol1924, xerr=ebv_heii_err_tol1924a, yerr=ebv_balmer_err_tol1924,
                       fmt='.', color=plotting_params['TOL_1924_416A']['color'])
ax_balmer_uv.scatter(ebv_uv_tol1924a, ebv_balmer_tol1924, s=150, color=plotting_params['TOL_1924_416A']['color'],
                      marker=plotting_params['TOL_1924_416A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_tol1924a, ebv_balmer_tol1924, xerr=ebv_uv_err_tol1924a, yerr=ebv_balmer_err_tol1924,
                       fmt='.', color=plotting_params['TOL_1924_416A']['color'])
ax_heii_uv.scatter(ebv_uv_tol1924a, ebv_heii_tol1924a, s=150, color=plotting_params['TOL_1924_416A']['color'],
                      marker=plotting_params['TOL_1924_416A']['marker'])
ax_heii_uv.errorbar(ebv_uv_tol1924a, ebv_heii_tol1924a, xerr=ebv_uv_err_tol1924a, yerr=ebv_heii_err_tol1924a,
                       fmt='.', color=plotting_params['TOL_1924_416A']['color'])


ax_balmer_heii.scatter(ebv_heii_tol89a, ebv_balmer_tol89, s=150, color=plotting_params['TOL_89A']['color'],
                      marker=plotting_params['TOL_89A']['marker'])
ax_balmer_heii.errorbar(ebv_heii_tol89a, ebv_balmer_tol89, xerr=ebv_heii_err_tol89a, yerr=ebv_balmer_err_tol89,
                       fmt='.', color=plotting_params['TOL_89A']['color'])
ax_balmer_uv.scatter(ebv_uv_tol89a, ebv_balmer_tol89, s=150, color=plotting_params['TOL_89A']['color'],
                      marker=plotting_params['TOL_89A']['marker'])
ax_balmer_uv.errorbar(ebv_uv_tol89a, ebv_balmer_tol89, xerr=ebv_uv_err_tol89a, yerr=ebv_balmer_err_tol89,
                       fmt='.', color=plotting_params['TOL_89A']['color'])
ax_heii_uv.scatter(ebv_uv_tol89a, ebv_heii_tol89a, s=150, color=plotting_params['TOL_89A']['color'],
                      marker=plotting_params['TOL_89A']['marker'])
ax_heii_uv.errorbar(ebv_uv_tol89a, ebv_heii_tol89a, xerr=ebv_uv_err_tol89a, yerr=ebv_heii_err_tol89a,
                       fmt='.', color=plotting_params['TOL_89A']['color'])

legend_targets = ['HE_2_10A', 'MRK_33A', 'NGC_3049A', 'NGC_3125A', 'NGC_4214A', 'NGC_4670A', 'TOL_1924_416A', 'TOL_89A']

for legend_name in legend_targets:
    target_name = str(legend_name[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')
    ax_blank.scatter([], [], s=150, marker=plotting_params[legend_name]['marker'],
                  color=plotting_params[legend_name]['color'], label=target_name)
ax_blank.legend(frameon=False, loc=3, bbox_to_anchor=[0.2, 0.1], fontsize=fontsize + 5)

dummy_x_data = np.linspace(-0.1, 0.5)
dummy_y_data = 1 * dummy_x_data
print('dummy_x_data ', dummy_x_data)
print('dummy_y_data ', dummy_y_data)
ax_balmer_heii.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)
ax_balmer_uv.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)
ax_heii_uv.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)

ax_balmer_heii.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_balmer_uv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_heii_uv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)

ax_balmer_heii.set_xlim(-0.15, 0.55)
ax_balmer_uv.set_xlim(-0.15, 0.55)
ax_heii_uv.set_xlim(-0.15, 0.55)
ax_balmer_heii.set_ylim(-0.15, 0.55)
ax_balmer_uv.set_ylim(-0.15, 0.55)
ax_heii_uv.set_ylim(-0.15, 0.55)



ax_balmer_uv.set_xticklabels([])
ax_balmer_uv.set_yticklabels([])

ax_balmer_heii.set_xlabel(r'E(B-V)$_{\rm HeII}$', fontsize=fontsize+5)
ax_balmer_heii.set_ylabel(r'E(B-V)$_{\rm Balmer}$', fontsize=fontsize+5)
ax_heii_uv.set_ylabel(r'E(B-V)$_{\rm HeII}$', fontsize=fontsize+5)
ax_heii_uv.set_xlabel(r'E(B-V)$_{\rm UV}$', fontsize=fontsize+5)

plt.savefig('plot_output/compare_ebv.png')


exit()



for target in plotting_params.keys():
    ebv_uv, ebv_uv_err, ebv_heii, ebv_heii_err, ebv_balmer, ebv_balmer_err = get_ebv(target[:-1], ext=target[-1])

    if plotting_params[target]['heii_1640_det'] & plotting_params[target]['heii_4686_det']:

        ax_ebv.errorbar(ebv_balmer, ebv_heii, xerr=ebv_balmer_err, yerr=ebv_heii_err,
                     fmt='.', color=plotting_params[target]['color'])
        ax_ebv.scatter(ebv_balmer, ebv_heii, s=150, color=plotting_params[target]['color'],
                      marker=plotting_params[target]['marker'])

legend_targets = ['HE_2_10A', 'MRK_33A', 'NGC_3049A', 'NGC_3125A', 'NGC_4214A', 'NGC_4670A', 'TOL_1924_416A', 'TOL_89A']

for legend_name in legend_targets:
    target_name = str(legend_name[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')
    ax_ebv.scatter([], [], s=150, marker=plotting_params[legend_name]['marker'],
                  color=plotting_params[legend_name]['color'], label=target_name)


ax_ebv.legend(frameon=False, fontsize=fontsize)

dummy_x_data = np.linspace(-0.1, 0.5)
dummy_y_data = 1 * dummy_x_data
print('dummy_x_data ', dummy_x_data)
print('dummy_y_data ', dummy_y_data)
ax_ebv.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)

plt.show()

exit()



# plot comparison
for target_id in plotting_params.keys():

    # get data
    beta = fit_dict['beta']
    beta_err = fit_dict['beta_err']
    line_flux_f1640 = fit_dict['line_flux_f1640']
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686']
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']

    mean_err_f1640 = np.mean(line_flux_err_f1640)
    mean_err_f4686 = np.mean(line_flux_err_f4686)

    he_line_ratio = line_flux_f1640 / line_flux_f4686
    he_line_ratio_err = np.sqrt((mean_err_f1640 / line_flux_f1640) ** 2 +
                                ((mean_err_f4686 * line_flux_f1640) / (line_flux_f4686 ** 2)) ** 2)
    ebv_uv, ebv_uv_err = E_BV_uv(beta=beta, err=beta_err)
    ebv_heii, ebv_heii_err = E_BV_ratio(observed_ratio=he_line_ratio, line_ratio='heII', err=he_line_ratio_err, reddening_law='R15')

    if (line_flux_f1640 / mean_err_f1640 > 3) & (line_flux_f4686 / mean_err_f4686 > 3):
        ax_ebv.errorbar(ebv_uv, ebv_heii, xerr=ebv_uv_err, yerr=ebv_heii_err,
                     fmt='.', color=plotting_params[target_id]['color'])
        ax_ebv.scatter(ebv_uv, ebv_heii, s=150, color=plotting_params[target_id]['color'],
                      marker=plotting_params[target_id]['marker'])

legend_targets = ['HE_2_10A', 'MRK_33A', 'NGC_3049A', 'NGC_3125A', 'NGC_4214A', 'NGC_4670A', 'TOL_1924_416A', 'TOL_89A']

for legend_name in legend_targets:
    target_name = str(legend_name[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')
    ax_ebv.scatter([], [], s=150, marker=plotting_params[legend_name]['marker'],
                  color=plotting_params[legend_name]['color'], label=target_name)


ax_ebv.legend(frameon=False, fontsize=fontsize)

dummy_x_data = np.linspace(-0.1, 0.5)
dummy_y_data = 1 * dummy_x_data
print('dummy_x_data ', dummy_x_data)
print('dummy_y_data ', dummy_y_data)
ax_ebv.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)


ax_ebv.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_ebv.set_xlabel(r'E(B-V)$_{\rm UV slope}$', fontsize=fontsize)
ax_ebv.set_ylabel(r'E(B-V)$_{\rm HeII}$', labelpad=-3, fontsize=fontsize)
plt.savefig('plot_output/ebv.png')
plt.cla()

