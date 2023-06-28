import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from photometry_tools import helper_func, plotting_tools
import os
import sep
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle

import aplpy

def subtract_background(img):
    # compute background
    bkg = sep.Background(np.array(img, dtype=float))
    # print('bkg.globalback ', bkg.globalback)
    # subtract a global background estimation
    img = img - bkg.globalback
    return img


hdu_hst_1 = fits.open('data/HLADATA-123426144353/hst_06639_03_wfpc2_f439w_pc/hst_06639_03_wfpc2_f439w_pc_drz.fits')


print(hdu_hst_1.info())

data_hst_1 = hdu_hst_1[1].data
data_hst_1 = subtract_background(img=data_hst_1)
wcs_hst_1 = WCS(hdu_hst_1[1].header)


print('wcs_hst_1 ', wcs_hst_1)
coords_slit = SkyCoord('12h45m17.44s +27d07m31.8s', unit=(u.hourangle, u.deg))



cutout_hst_1 = helper_func.get_img_cutout(img=data_hst_1, wcs=wcs_hst_1,
                                        coord=coords_slit, cutout_size=(50,50))



figure = plt.figure(figsize=(15, 15))
fontsize = 23
ax_hst_1 = figure.add_axes([0.06, 0.04, 0.93, 0.93], projection=cutout_hst_1.wcs)


# ax_hst_1.imshow(cutout_hst_1.data, origin='lower', vmin=0.0, vmax=50)
ax_hst_1.imshow(np.log10(cutout_hst_1.data))


coords_sdss_1 = SkyCoord(ra=191.320310*u.deg, dec=27.125218*u.deg)
coords_sdss_2 = SkyCoord(ra=191.321910*u.deg, dec=27.125597*u.deg)

circle_hst = SphericalCircle(coords_sdss_1, 1.5 * u.arcsec, edgecolor='r', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst_1.get_transform('fk5'))
ax_hst_1.add_patch(circle_hst)


circle_hst = SphericalCircle(coords_sdss_2, 1.5 * u.arcsec, edgecolor='b', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst_1.get_transform('fk5'))
ax_hst_1.add_patch(circle_hst)




ax_hst_1.text(cutout_hst_1.data.shape[0] * 0.05, cutout_hst_1.data.shape[1] * 0.95, 'F439W', color='w', fontsize=fontsize)



plotting_tools.arr_axis_params(ax=ax_hst_1, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)

#
# overlay = ax_hst_1.get_coords_overlay('fk5')
# overlay.grid(color='white', ls='dotted')
# overlay[0].set_axislabel('Right Ascension (J2000)')
# overlay[1].set_axislabel('Declination (J2000)')
#

# plt.show()

plt.savefig('plot_output/HST_overview.png')


exit()

# from os import listdir
#
#
# print(hdu_fors2.info())
#
# print(hdu_fors2[1].data.names)
#
# for key in hdu_fors2[1].header.keys():
#     print(key, ' ', hdu_fors2[1].header[key])


# print(hdu_fors2[1].data['WAVE'])
# print(hdu_fors2[1].data['FLUX'])




plt.show()

