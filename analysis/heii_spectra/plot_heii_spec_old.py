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
        'ylim_g140l': (-0.25*1e-14, 1.8*1e-14)},
    'HE_2_10B': {
        'ylim_g140l': (-0.25*1e-14, 1.5*1e-14)},
    'HE_2_10C': {
        'ylim_g140l': (-0.25*1e-14, 1.05*1e-14)},
    'HE_2_10D': {
        'ylim_g140l': (-0.25*1e-14, 0.8*1e-14)},
    'MRK_33A': {
        'ylim_g140l': (-0.15*1e-14, 0.7*1e-14)},
    'MRK_33B': {
        'ylim_g140l': (-0.15*1e-14, 0.7*1e-14)},
    'NGC_3049A': {
        'ylim_g140l': (-0.15*1e-14, 2.4*1e-14)},
    'NGC_3049B': {
        'ylim_g140l': (-0.15*1e-14, 0.3*1e-14)},
    'NGC_3125A': {
        'ylim_g140l': (-0.25*1e-14, 1.8*1e-14)},
    'NGC_4214A': {
        'ylim_g140l': (-0.25*1e-14, 9.8*1e-14)},
    'NGC_4670A': {
        'ylim_g140l': (-0.25*1e-14, 1.8*1e-14)},
    'NGC_4670B': {
        'ylim_g140l': (-0.25*1e-14, 0.7*1e-14)},
    'NGC_4670C': {
        'ylim_g140l': (-0.25*1e-14, 0.7*1e-14)},
    'TOL_1924_416A': {
        'ylim_g140l': (-0.25*1e-14, 3.8*1e-14)},
    'TOL_1924_416B': {
        'ylim_g140l': (-0.15*1e-14, 0.7*1e-14)},
    'TOL_89A': {
        'ylim_g140l': (-0.25*1e-14, 1.8*1e-14)}
}



for target_id in plotting_params.keys():
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

    # get data
    continuum_g140l = fit_dict['continuum_g140l']
    wave_g140l = fit_dict['wave_g140l']
    flux_g140l = fit_dict['flux_g140l']
    flux_err_g140l = fit_dict['flux_err_g140l']
    continuum_g430m = fit_dict['continuum_g430m']
    wave_g430m = fit_dict['wave_g430m']
    flux_g430m = fit_dict['flux_g430m']
    flux_err_g430m = fit_dict['flux_err_g430m']
    beta = fit_dict['beta']
    beta_err = fit_dict['beta_err']
    line_flux_f1640 = fit_dict['line_flux_f1640']
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686']
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']

    ylim_g140l = plotting_params[target_id]['ylim_g140l']
    xlim_f1640 = (1640 - 50, 1640 + 50)

    # create plotting
    figure = plt.figure(figsize=(25, 7))
    fontsize = 17
    ax_g140l = figure.add_axes([0.035, 0.08, 0.66, 0.88])
    ax_g430m = figure.add_axes([0.72, 0.08, 0.25, 0.88])
    ax_f1640 = figure.add_axes([0.49, 0.55, 0.18, 0.42])

    # plot g140l
    ax_g140l.step(wave_g140l, flux_g140l, where='mid', color='k', linewidth=2)
    for window in calzetti_windows:
        mask_include_fit = (wave_g140l > window[0]) & (wave_g140l < window[1])
        ax_g140l.step(wave_g140l[mask_include_fit], flux_g140l[mask_include_fit], where='mid', color='tab:red', linewidth=2)
    ax_g140l.fill_between(wave_g140l, flux_g140l-flux_err_g140l, flux_g140l+flux_err_g140l,
                          color='gray', alpha=0.7)
    ax_g140l.plot([], [], color='white', label='G140L')
    ax_g140l.plot(wave_g140l, continuum_g140l, linestyle='--', linewidth=2, color='tab:blue', label=r'$\beta = %.2f \pm %.2f$' % (beta, beta_err))
    in_window = (wave_g140l > 1630) & (wave_g140l < 1652)
    ax_g140l.step(wave_g140l[in_window], flux_g140l[in_window], c='cyan', lw=2)

    # plot f1640
    mask_wave_f1640 = (wave_g140l > xlim_f1640[0]) & (wave_g140l < xlim_f1640[1])
    ax_f1640.step(wave_g140l[mask_wave_f1640], flux_g140l[mask_wave_f1640], where='mid', color='k', linewidth=2)
    ax_f1640.fill_between(wave_g140l[mask_wave_f1640], (flux_g140l-flux_err_g140l)[mask_wave_f1640],
                          (flux_g140l+flux_err_g140l)[mask_wave_f1640], color='gray', alpha=0.7)
    ax_f1640.plot(wave_g140l[mask_wave_f1640], continuum_g140l[mask_wave_f1640], linestyle='--', linewidth=2, color='tab:blue')
    ax_f1640.step(wave_g140l[in_window], flux_g140l[in_window], c='cyan', lw=2)

    # plot g430m
    ax_g430m.step(wave_g430m, flux_g430m, where='mid', color='k', linewidth=2)
    ax_g430m.fill_between(wave_g430m, flux_g430m-flux_err_g430m, flux_g430m+flux_err_g430m,
                          color='gray', alpha=0.7)
    ax_g430m.plot(wave_g430m, continuum_g430m, linestyle='--', linewidth=2, color='tab:blue')
    if target_id == 'MRK_33B':
        # in_window = (wave_g430m > 4670) & (wave_g430m < 4705)
        in_window = ((wave_g430m >= 4670) & (wave_g430m <= 4676) |
                     (wave_g430m >= 4678.5) & (wave_g430m <= 4705))


    else:
        in_window = (wave_g430m > 4670) & (wave_g430m < 4705)

    ax_g430m.step(wave_g430m[in_window], flux_g430m[in_window], c='cyan', lw=2)

    # set limits
    ax_g140l.set_xlim(1150, 1700)
    ax_g140l.set_ylim(ylim_g140l)

    mean_flux_f1640 = np.nanmean(flux_g140l[mask_wave_f1640])
    std_flux_f1640 = np.nanstd(flux_g140l[mask_wave_f1640])
    min_flux_f1640 = np.nanmin(flux_g140l[mask_wave_f1640])
    max_flux_f1640 = np.nanmax(flux_g140l[mask_wave_f1640])
    max_continuum_f1640 = np.nanmax(continuum_g140l[mask_wave_f1640])
    ax_f1640.set_ylim(min_flux_f1640 - std_flux_f1640, np.max([max_flux_f1640, max_continuum_f1640]) + std_flux_f1640)

    xlim_g430m = (4686 - 100, 4686 + 100)
    ax_g430m.set_xlim(xlim_g430m)
    mask_wave_g430m = (wave_g430m > xlim_g430m[0]) & (wave_g430m < xlim_g430m[1])
    mean_flux_g430m = np.nanmean(flux_g430m[mask_wave_g430m])
    std_flux_g430m = np.nanstd(flux_g430m[mask_wave_g430m])
    min_flux_g430m = np.nanmin(flux_g430m[mask_wave_g430m])
    max_flux_g430m = np.nanmax(flux_g430m[mask_wave_g430m])
    max_continuum_g430m = np.nanmax(continuum_g430m[mask_wave_g430m])
    ax_g430m.set_ylim(min_flux_g430m - 1*std_flux_g430m, np.max([max_flux_g430m, max_continuum_g430m]) + 3*std_flux_g430m)

    ax_g140l.fill_between([1630, 1652], ax_g140l.get_ylim()[1], ax_g140l.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")
    ax_f1640.fill_between([1630, 1652], ax_f1640.get_ylim()[1], ax_f1640.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")
    ax_g430m.fill_between([4670, 4705], ax_g430m.get_ylim()[1], ax_g430m.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")

    flux_str_f1640 = r'HeII(1640) = %.2f $^{+%.2f}_{-%.2f} \times 10^{-15}$ erg/s/cm$^{2}$ ' % \
                     (line_flux_f1640, line_flux_err_f1640[0], line_flux_err_f1640[1])
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

    ax_g140l.set_title(target_id, fontsize=fontsize)

    ax_g140l.set_xlabel('Rest Wavelength (Å)', labelpad=-3, fontsize=fontsize)
    ax_g430m.set_xlabel('Rest Wavelength (Å)', labelpad=-3, fontsize=fontsize)
    ax_g140l.set_ylabel('Flux (erg/s/cm/cm/Å)', labelpad=-15, fontsize=fontsize)

    plt.savefig('plot_output/spec_result_%s.png' % target_id)
    plt.cla()


exit()

