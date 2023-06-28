import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from os import path

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins
from plotbin.plot_velfield import plot_velfield

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import ConnectionPatch

from image_registration import chi2_shift
from image_registration.fft_tools import shift

import glob
import os
import shutil

from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astroquery.mast import Observations
from drizzlepac import tweakreg
from drizzlepac import astrodrizzle


C = 299792.458  # speed of light in km/s
# from scipy.constants import c as speed_of_light

cutout_size = (30, 30)

file_name_he2_10_muse = '/home/benutzer/data/observation/muse_data/he2_10/ADP.2016-06-17T18 13 44.227.fits'
hdu_muse = pyfits.open(file_name_he2_10_muse)
head_muse = hdu_muse[1].header
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"

# mask_h_alpha = (wave_muse > 6560) & (wave_muse < 6610)
mask_h_alpha = (wave_muse > 6560) & (wave_muse < 6610)

print(wave_muse)
plt.scatter(wave_muse, cube_muse[:, 160, 170])
plt.show()
plt.imshow(np.nansum(cube_muse[mask_h_alpha, :, :], 0))
plt.show()
#

# file_name_he2_10_f555w = '/home/benutzer/data/observation/heII_paper/he2_10/f435w/hst_10609_54_acs_hrc_f435w_drz.fits'
file_name_he2_10_f555w = '/home/benutzer/data/observation/heII_paper/he2_10/f435w/hst_06580_02_wfpc2_f658n_pc_drz.fits'



hdu_hst_f555w = pyfits.open(file_name_he2_10_f555w)
data_f555w = hdu_hst_f555w[1].data
wcs_f555w = WCS(hdu_hst_f555w[1].header)

# save muse data in file


hdu = fits.PrimaryHDU(np.nansum(cube_muse[mask_h_alpha, :, :], 0), header=wcs_muse.celestial.to_header()) # This ensures the same header is written to the new file
hdulist = fits.HDUList([hdu])
hdulist.writeto('muse_img.fits', overwrite=True)

hdu = fits.PrimaryHDU(data_f555w, header=hdu_hst_f555w[1].header) # This ensures the same header is written to the new file
hdulist = fits.HDUList([hdu])
hdulist.writeto('hst_img.fits', overwrite=True)



tweakreg.TweakReg(['muse_img.fits', 'hst_img.fits'],
    conv_width=3.5,
    threshold=4000,
    searchrad=4.0,
    peakmax=70000,
    fitgeometry='rscale',
    configobj = None,
    interactive=False,
    shiftfile=True,
    outshifts='shift_searchrad.txt',
    updatehdr=True)

#reload muse img with wcs

hdu_new = pyfits.open('muse_img.fits')
head_new = hdu_new[0].header
# get new wcs for MUSE
wcs_muse = WCS(head_new)

hdu_new = pyfits.open('hst_img.fits')
head_new = hdu_new[0].header
# get new wcs for MUSE
wcs_f555w = WCS(head_new)


coords_a_hst = SkyCoord('8h36m15.095s -26d24m34.5s', unit=(u.hourangle, u.deg))
coords_b_hst = SkyCoord('8h36m15.059s -26d24m34.59s', unit=(u.hourangle, u.deg))
coords_c_hst = SkyCoord('8h36m15.030s -26d24m34.59s', unit=(u.hourangle, u.deg))
coords_d_hst = SkyCoord('8h36m15.008s -26d24m34.67s', unit=(u.hourangle, u.deg))

coords_1_muse = SkyCoord('8h36m15.14s -26d24m33.5s', unit=(u.hourangle, u.deg))
coords_2_muse = SkyCoord('8h36m15.06s -26d24m33.7s', unit=(u.hourangle, u.deg))


coords_cent = SkyCoord('8h36m15.095s -26d24m34.5s', unit=(u.hourangle, u.deg))
cutout_hst = helper_func.get_img_cutout(img=data_f555w, wcs=wcs_f555w,
                                        coord=coords_cent, cutout_size=cutout_size)

cutout_muse = helper_func.get_img_cutout(img=np.nansum(cube_muse[mask_h_alpha, :, :], 0), wcs=wcs_muse.celestial,
                                         coord=coords_cent, cutout_size=cutout_size)


figure = plt.figure(figsize=(30, 15))
fontsize = 23
ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=cutout_hst.wcs)
ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93], projection=cutout_muse.wcs)


ax_hst.imshow(np.log10(cutout_hst.data), cmap='bone')
ax_muse.imshow(np.log10(cutout_muse.data), cmap='Greys')


circle_hst = SphericalCircle(coords_a_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_b_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_c_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_d_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)

circle_muse = SphericalCircle(coords_a_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)
circle_muse = SphericalCircle(coords_b_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)
circle_muse = SphericalCircle(coords_c_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)
circle_muse = SphericalCircle(coords_d_hst, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)



pix_a = cutout_hst.wcs.world_to_pixel(coords_a_hst)
ax_hst.text(pix_a[0] - 3, pix_a[1] + 3, 'A', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
pix_b = cutout_hst.wcs.world_to_pixel(coords_b_hst)
ax_hst.text(pix_b[0] + 1, pix_b[1] + 3, 'B', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
pix_c = cutout_hst.wcs.world_to_pixel(coords_c_hst)
ax_hst.text(pix_c[0] - 0, pix_c[1] + 3, 'C', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
pix_d = cutout_hst.wcs.world_to_pixel(coords_d_hst)
ax_hst.text(pix_d[0] + 1, pix_d[1] + 3, 'D', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)


# circle_muse = SphericalCircle(coords_1_muse,0.5 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
#                                alpha=1.0, transform=ax_muse.get_transform('fk5'))
# ax_muse.add_patch(circle_muse)
# circle_muse = SphericalCircle(coords_2_muse,0.5 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
#                                alpha=1.0, transform=ax_muse.get_transform('fk5'))
# ax_muse.add_patch(circle_muse)
#
# pix_a = cutout_muse.wcs.world_to_pixel(coords_1_muse)
# ax_muse.text(pix_a[0] - 0, pix_a[1] + 3, '1', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
# pix_b = cutout_muse.wcs.world_to_pixel(coords_2_muse)
# ax_muse.text(pix_b[0] - 0, pix_b[1] + 3, '2', horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)
#

ax_hst.text(cutout_hst.data.shape[0]*0.95, cutout_hst.data.shape[1]*0.95, 'HST F435W', horizontalalignment='right', color='r', fontsize=fontsize)
ax_muse.text(cutout_muse.data.shape[0]*0.95, cutout_muse.data.shape[1]*0.95, 'MUSE', horizontalalignment='right', color='r', fontsize=fontsize)

ax_hst.set_title('HE 2-10', fontsize=fontsize)

plotting_tools.arr_axis_params(ax=ax_hst, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)

plotting_tools.arr_axis_params(ax=ax_muse, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)

# plt.show()
plt.savefig('plot_output/hst_muse_overview.png')


