import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
import multicolorfits as mcf
from astropy.wcs.utils import fit_wcs_from_points


file_name_tol_1924_416_muse = '/home/benutzer/data/observation/heII_paper/tol1924_416/muse/ADP.2017-03-23T15 16 34.777.fits'
hdu_muse = fits.open(file_name_tol_1924_416_muse)
head_muse = hdu_muse[1].header
print(head_muse)
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"
mask_h_alpha = (wave_muse > 6620) & (wave_muse < 6630)
data_muse_h_alpha = np.nansum(cube_muse[mask_h_alpha, :, :], 0)


file_name_tol_1924_416_f336w = '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f336w_pc_drz.fits'
file_name_tol_1924_416_f555w = '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f555w_pc_drz.fits'
file_name_tol_1924_416_f814w = '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f814w_pc_drz.fits'
hdu_hst_f336w = fits.open(file_name_tol_1924_416_f336w)
hdu_hst_f555w = fits.open(file_name_tol_1924_416_f555w)
hdu_hst_f814w = fits.open(file_name_tol_1924_416_f814w)
data_f336w = hdu_hst_f336w[1].data
data_f555w = hdu_hst_f555w[1].data
data_f814w = hdu_hst_f814w[1].data
wcs_f336w = WCS(hdu_hst_f336w[1].header)
wcs_f555w = WCS(hdu_hst_f555w[1].header)
wcs_f814w = WCS(hdu_hst_f814w[1].header)

# # update MUSE
# stars = [np.array([93.013, 215.373, 72.02]), np.array([329.953, 164.676, 291.04])]
# known_coords = SkyCoord(['19h27m59.554s -41d34m01.89s', '19h27m57.359s -41d34m34.86s', '19h27m59.920s -41d34m09.72s'], unit=(u.hourangle, u.deg))
# new_muse_wcs = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_muse.celestial)
# # new_muse_wcs = wcs_muse.celestial

# update MUSE
stars = [np.array([215.373, 236.41, 219.05]), np.array([164.676, 143.91, 207.00])]
known_coords = SkyCoord(['19h27m57.338s -41d34m35.63s', '19h27m56.970s -41d34m39.72s', '19h27m57.276s -41d34m27.21s'], unit=(u.hourangle, u.deg))
new_muse_wcs = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_muse.celestial)
# new_muse_wcs = wcs_muse.celestial


# figure = plt.figure(figsize=(30, 15))
# fontsize = 23
# ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=wcs_f814w)
# ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93], projection=new_muse_wcs)
#
# ax_hst.imshow(np.log10(data_f814w), origin='lower')
# ax_muse.imshow(np.log10(np.nansum(cube_muse[:, :, :], 0)), origin='lower')
# plt.show()
# exit()

cutout_size = (17, 17)
coords_cent = SkyCoord('19h27m58.182s -41d34m33.94s', unit=(u.hourangle, u.deg))
cutout_f336w = helper_func.get_img_cutout(img=data_f336w, wcs=wcs_f336w, coord=coords_cent, cutout_size=cutout_size)
cutout_f555w = helper_func.get_img_cutout(img=data_f555w, wcs=wcs_f555w, coord=coords_cent, cutout_size=cutout_size)
cutout_f814w = helper_func.get_img_cutout(img=data_f814w, wcs=wcs_f814w, coord=coords_cent, cutout_size=cutout_size)
cutout_muse = helper_func.get_img_cutout(img=data_muse_h_alpha, wcs=new_muse_wcs, coord=coords_cent, cutout_size=cutout_size)


# # reproject hst_cutouts to f336w
new_wcs = cutout_f336w.wcs
new_shape = cutout_f336w.data.shape
hst_img_g = plotting_tools.reproject_image(data=cutout_f555w.data,
                                           wcs=cutout_f555w.wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)
hst_img_r = plotting_tools.reproject_image(data=cutout_f814w.data,
                                           wcs=cutout_f814w.wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)

grey_hst_r = mcf.greyRGBize_image(hst_img_r, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=2.2, checkscale=False)
grey_hst_g = mcf.greyRGBize_image(hst_img_g, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=2.2, checkscale=False)
grey_hst_b = mcf.greyRGBize_image(cutout_f336w.data, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=2.2, checkscale=False)
hst_r_purple = mcf.colorize_image(grey_hst_r, '#FF0000', colorintype='hex', gammacorr_color=2.2)
hst_g_orange = mcf.colorize_image(grey_hst_g, '#00FF00', colorintype='hex', gammacorr_color=2.2)
hst_b_blue = mcf.colorize_image(grey_hst_b, '#0000FF', colorintype='hex', gammacorr_color=2.2)
rgb_hst_image = mcf.combine_multicolor([hst_r_purple, hst_g_orange, hst_b_blue], gamma=2.2, inverse=False)


coords_a_hst = SkyCoord('19h27m58.382s -41d34m30.94s', unit=(u.hourangle, u.deg))

color_a = 'tab:red'


figure = plt.figure(figsize=(30, 15))
fontsize = 23
ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=new_wcs)
ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93], projection=cutout_muse.wcs)
ax_balmer_1 = figure.add_axes([0.31, 0.12, 0.29, 0.3])
ax_balmer_2 = figure.add_axes([0.61, 0.12, 0.29, 0.3])


ax_hst.imshow(rgb_hst_image)
ax_muse.imshow(np.log10(cutout_muse.data), cmap='Greys')


circle_hst = SphericalCircle(coords_a_hst, 0.5 * u.arcsec, edgecolor=color_a, facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)


circle_muse = SphericalCircle(coords_a_hst, 0.5 * u.arcsec, edgecolor=color_a, facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)



pix_a = new_wcs.world_to_pixel(coords_a_hst)
ax_hst.text(pix_a[0] - 8, pix_a[1] + 8, 'A', color=color_a, horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)

pix_a = cutout_muse.wcs.world_to_pixel(coords_a_hst)
ax_muse.text(pix_a[0] - 2.5, pix_a[1] + 2.5, 'A', color=color_a, horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)


ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.96, 'HST', horizontalalignment='left', color='white', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.93, 'F336W', horizontalalignment='left', color='b', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.90, 'F555W', horizontalalignment='left', color='g', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.87, 'F814W', horizontalalignment='left', color='r', fontsize=fontsize)

ax_muse.text(cutout_muse.data.shape[0]*0.03, cutout_muse.data.shape[1]*0.95, r'MUSE H$\alpha$', horizontalalignment='left', color='k', fontsize=fontsize)

ax_hst.set_title('TOL 1924-416', fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_hst, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)

plotting_tools.arr_axis_params(ax=ax_muse, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)


balmer_dict_a = np.load('data_output/balmer_dict_TOL_1924_416.npy', allow_pickle=True).item()

ax_balmer_1.plot(balmer_dict_a['wavelength'], balmer_dict_a['total_flux'] * 1e-5, color=color_a)
ax_balmer_2.plot(balmer_dict_a['wavelength'], balmer_dict_a['total_flux'] * 1e-5, color=color_a)

ax_balmer_1.plot([4863 + 44.5, 4863 + 44.5], [-2, 28], linestyle='--', color='k')
ax_balmer_1.plot([4960 + 46, 4960 + 46], [-2, 28], linestyle='--', color='k')
ax_balmer_1.plot([5008 + 46, 5008 + 46], [-2, 28], linestyle='--', color='k')

ax_balmer_2.plot([6550 + 60, 6550 + 60], [-2, 28], linestyle='--', color='k')
ax_balmer_2.plot([6565 + 60, 6565 + 60], [-2, 28], linestyle='--', color='k')
ax_balmer_2.plot([6585 + 60, 6585 + 60], [-2, 28], linestyle='--', color='k')


ax_balmer_1.text(4863 + 2 + 44.5, 12.6, r'H$\beta$', fontsize=fontsize)
ax_balmer_1.text(4960 + 2 + 46, 12.6, r'[OIII]', fontsize=fontsize)
ax_balmer_1.text(5008 + 2 + 46, 12.6, r'[OIII]', fontsize=fontsize)

ax_balmer_2.text(6550 + 2 + 60, 12.6, r'[NII]', fontsize=fontsize)
ax_balmer_2.text(6565 + 2 + 60, 12.6, r'H$\alpha$', fontsize=fontsize)
ax_balmer_2.text(6585 + 2 + 60, 12.6, r'[NII]', fontsize=fontsize)

ax_balmer_1.set_xlim(4886, 5074)
ax_balmer_2.set_xlim(6600, 6660)

ax_balmer_1.set_ylim(-0.2, 13.6)
ax_balmer_2.set_ylim(-0.2, 13.6)


# hide the spines between ax and ax2
ax_balmer_1.spines['right'].set_visible(False)
ax_balmer_2.spines['left'].set_visible(False)
# ax_balmer_1.yaxis.tick_left()
# ax_balmer_1.tick_params(labelright='off')
ax_balmer_2.yaxis.tick_right()

ax_balmer_1.set_xticks([4900, 4925, 4950, 4975, 5000])
ax_balmer_1.set_ylabel('Flux [10$^{-15}$ erg/s/cm/cm/Å]', color='white', fontsize=fontsize)
ax_balmer_1.set_xlabel('Observed Wavelength [Å]                       ', color='white', fontsize=fontsize)

ax_balmer_2.set_xlabel('Observed Wavelength [Å]', fontsize=fontsize)

# ax_balmer_1.spines['left'].set_color('white')
ax_balmer_1.tick_params(axis='both', colors="white", labelsize=fontsize)
ax_balmer_2.tick_params(axis='both', colors="black", labelsize=fontsize)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax_balmer_1.transAxes, color='k', clip_on=False)
ax_balmer_1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax_balmer_1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax_balmer_2.transAxes)  # switch to the bottom axes
ax_balmer_2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax_balmer_2.plot((-d, +d), (-d, +d), **kwargs)


# plt.show()
plt.savefig('plot_output/hst_muse_overview.png')


