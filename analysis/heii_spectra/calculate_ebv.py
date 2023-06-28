import os , glob
import numpy as np
import pandas as pd
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


# print table
for target_id in plotting_params.keys():
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

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

    target_name = str(target_id[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')
    track = str(target_id[-1])
    if track != 'A':
        target_name = ''

    name_str = '%s & %s & ' % (target_name, track)
    f1640_str = '%.1f$\\pm$%.1f & ' % (line_flux_f1640, mean_err_f1640)
    f4686_str = '%.1f$\\pm$%.1f & ' % (line_flux_f4686, mean_err_f4686)

    if (plotting_params[target_id]['heii_1640_det']) & (plotting_params[target_id]['heii_4686_det']):
        he_str = '%.1f$\\pm$%.1f & ' % (he_line_ratio, he_line_ratio_err)
        ebv_he_str = '%.2f$\\pm$%.2f & ' % (ebv_heii, ebv_heii_err)
    else:
        he_str = '-- & '
        ebv_he_str = '-- & '
    beta_str = '%.1f$\\pm$%.1f & ' % (beta, beta_err)
    ebv_beta_str = '%.2f$\\pm$%.2f \\\\ ' % (ebv_uv, ebv_uv_err)
    print(name_str + f1640_str + f4686_str + he_str + beta_str + ebv_he_str + ebv_beta_str)

figure = plt.figure(figsize=(13, 10))
fontsize = 17
ax_he = figure.add_axes([0.08, 0.07, 0.91, 0.92])

# plot comparison
for target_id in plotting_params.keys():
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

    # get data
    line_flux_f1640 = fit_dict['line_flux_f1640'] * 1e-15
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686'] * 1e-15
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']

    mean_err_f1640 = np.mean(line_flux_err_f1640) * 1e-15
    mean_err_f4686 = np.mean(line_flux_err_f4686) * 1e-15
    if (line_flux_f1640 / mean_err_f1640 > 3) & (line_flux_f4686 / mean_err_f4686 > 3):
        ax_he.errorbar(line_flux_f1640, line_flux_f4686, xerr=mean_err_f1640, yerr=mean_err_f4686,
                     fmt='.', color=plotting_params[target_id]['color'])
        ax_he.scatter(line_flux_f1640, line_flux_f4686, s=150, color=plotting_params[target_id]['color'],
                      marker=plotting_params[target_id]['marker'])

legend_targets = ['HE_2_10A', 'MRK_33A', 'NGC_3049A', 'NGC_3125A', 'NGC_4214A', 'NGC_4670A', 'TOL_1924_416A', 'TOL_89A']


for legend_name in legend_targets:
    target_name = str(legend_name[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')

    ax_he.scatter([], [], s=150, marker=plotting_params[legend_name]['marker'],
                  color=plotting_params[legend_name]['color'], label=target_name)


ax_he.legend(frameon=False, fontsize=fontsize)

dummy_x_data = np.linspace(2e-15, 4e-14)
# dummy_y_data = 1 * dummy_x_data
dummy_y_data = dummy_x_data - np.log10(0.89)

print('dummy_x_data ', dummy_x_data)
print('dummy_y_data ', dummy_y_data)

ax_he.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)

ax_he.text(3e-15, 3e-15, 'Slope 0.89', rotation=38, fontsize=fontsize)

ax_he.set_xscale('log')
ax_he.set_yscale('log')
ax_he.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_he.set_xlabel(r'Flux HeII (1640) [erg s$^{-1}$ cm$^{-2}$]', fontsize=fontsize)
ax_he.set_ylabel(r'Flux HeII (4686) [erg s$^{-1}$ cm$^{-2}$]', labelpad=-3, fontsize=fontsize)
plt.savefig('plot_output/he_flux.png')
plt.cla()





figure = plt.figure(figsize=(13, 10))
fontsize = 17
ax_he = figure.add_axes([0.08, 0.07, 0.91, 0.92])

# plot comparison
for target_id in plotting_params.keys():
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

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
        ax_he.errorbar(ebv_uv, ebv_heii, xerr=ebv_uv_err, yerr=ebv_heii_err,
                     fmt='.', color=plotting_params[target_id]['color'])
        ax_he.scatter(ebv_uv, ebv_heii, s=150, color=plotting_params[target_id]['color'],
                      marker=plotting_params[target_id]['marker'])

legend_targets = ['HE_2_10A', 'MRK_33A', 'NGC_3049A', 'NGC_3125A', 'NGC_4214A', 'NGC_4670A', 'TOL_1924_416A', 'TOL_89A']

for legend_name in legend_targets:
    target_name = str(legend_name[:-1]).replace('_', '-')
    target_name = target_name.replace('NGC-', 'NGC ')
    target_name = target_name.replace('MRK-', 'MRK ')
    target_name = target_name.replace('HE-', 'HE ')
    target_name = target_name.replace('TOL-', 'TOL ')
    ax_he.scatter([], [], s=150, marker=plotting_params[legend_name]['marker'],
                  color=plotting_params[legend_name]['color'], label=target_name)


ax_he.legend(frameon=False, fontsize=fontsize)

dummy_x_data = np.linspace(-0.1, 0.5)
dummy_y_data = 1 * dummy_x_data
print('dummy_x_data ', dummy_x_data)
print('dummy_y_data ', dummy_y_data)
ax_he.plot(dummy_x_data, dummy_y_data, linestyle='--', color='k', linewidth=2)


ax_he.tick_params(axis='both', which='both', width=1.5, length=4, right=True, top=True, direction='in', labelsize=fontsize)
ax_he.set_xlabel(r'E(B-V)$_{\rm UV slope}$', fontsize=fontsize)
ax_he.set_ylabel(r'E(B-V)$_{\rm HeII}$', labelpad=-3, fontsize=fontsize)
plt.savefig('plot_output/ebv.png')
plt.cla()

